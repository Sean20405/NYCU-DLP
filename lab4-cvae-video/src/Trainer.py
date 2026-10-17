import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timezone, timedelta

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def Generate_PSNR_per_frame(imgs1, imgs2, data_range=1.):
    """Per-frame PSNR averaged over batch for tensors with shape (B, T, C, H, W)."""
    mse_per_frame = ((imgs1 - imgs2) ** 2).mean(dim=(2, 3, 4))
    mse_per_frame = torch.clamp(mse_per_frame, min=1e-10)
    psnr_per_frame = 20 * log10(data_range) - 10 * torch.log10(mse_per_frame)
    return psnr_per_frame.mean(dim=0)


def kl_criterion(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD / (mu.size(0) * mu.size(1))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Pixel-wise MSE
        mse = (inputs - targets) ** 2
        
        # Focal weight: pixels with higher error get higher weight
        # weight = |diff|^gamma
        weight = torch.abs(inputs - targets) ** self.gamma
        
        focal_loss = self.alpha * weight * mse

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class kl_annealing():
    def __init__(self, args, current_epoch=0):  # TODO
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.beta_min = args.kl_beta_min
        self.current_epoch = current_epoch
        self.num_epoch = args.num_epoch
        if self.type == "cyc":
            self.betas = self.frange_cycle_linear(
                n_iter=self.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=self.cycle,
                ratio=self.ratio
            )
        elif self.type == "mono":
            self.betas = self.frange_cycle_linear(
                n_iter=self.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=1,
                ratio=self.ratio
            )
        else:
            self.betas = [1.0] * self.num_epoch
        
    def update(self):  # TODO
        self.current_epoch += 1
    
    def get_beta(self):  # TODO
        idx = min(self.current_epoch, self.num_epoch - 1)
        return max(float(self.betas[idx]), self.beta_min)

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):  # TODO
        betas = np.ones(n_iter)
        period = n_iter // n_cycle
        step = (stop - start) / (period * ratio)

        for i in range(n_iter):
            i_in_cycle = i % period
            if i_in_cycle < period * ratio:
                betas[i] = start + step * i_in_cycle
            else:
                betas[i] = stop
        
        return betas
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        if self.args.optim == "Adam":
            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-2)
        
        # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch)

        # Set warmup scheduler to pass first beta annealing cycle to warm up the model, avoid large lr at the beginning
        # warmup_epochs = 2.5 * args.num_epoch // args.kl_anneal_cycle if args.kl_anneal_cycle > 0 else 5
        # start_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=warmup_epochs, eta_min=args.lr * 0.05)
        # main_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch - warmup_epochs, eta_min=1e-6)
        # self.scheduler  = optim.lr_scheduler.SequentialLR(self.optim, schedulers=[start_scheduler, main_scheduler], milestones=[warmup_epochs])

        self.kl_annealing = kl_annealing(args, current_epoch=0)

        if args.loss_type == 'l1':
            self.mse_criterion = nn.L1Loss()
        elif args.loss_type == 'mse':
            self.mse_criterion = nn.MSELoss()
        elif args.loss_type == 'focal':
            self.mse_criterion = FocalLoss()
        
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.train_vi_stride = args.train_vi_stride
        self.batch_size = args.batch_size
        self.tail_weight_scale = args.tail_weight_scale
        self.tail_weight_power = args.tail_weight_power

        if args.test: # Load existing tensorboard logs
            self.twriter = SummaryWriter(log_dir=args.logdir)
        else:
            # Tensorboard
            time_str = datetime.now(timezone(timedelta(hours=8))).strftime("%b%d_%H%M%S")
            log_dir = os.path.join(args.log_root, f"{args.exp_name}_{time_str}")
            self.twriter = SummaryWriter(log_dir=log_dir)

            # Create folder
            os.makedirs(os.path.join(args.save_root, args.exp_name), exist_ok=True)
        
    def forward(self, img, label):
        pass



    def reconstruction_loss(self, pred, target):
        if self.tail_weight_scale <= 0 or pred.size(1) <= 1:
            return self.mse_criterion(pred, target)

        T = pred.size(1)
        step_pos = torch.linspace(0.0, 1.0, steps=T, device=pred.device, dtype=pred.dtype)
        weights = 1.0 + self.tail_weight_scale * (step_pos ** self.tail_weight_power)
        weights = weights / weights.sum()

        per_frame_losses = []
        for t in range(T):
            per_frame_losses.append(self.mse_criterion(pred[:, t], target[:, t]))
        per_frame_losses = torch.stack(per_frame_losses)
        return torch.sum(per_frame_losses * weights)


    def training_stage(self):
        best_psnr = -1
        val_loader = self.val_dataloader()

        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss_dict = self.training_one_step(img, label, adapt_TeacherForcing)
                loss = loss_dict['total_loss']
                
                # Tensorboard logging (step)
                step = self.current_epoch * len(train_loader) + pbar.n
                self.twriter.add_scalar('Loss/train/total', loss.item(), step)
                self.twriter.add_scalar('Loss/train/mse', loss_dict['mse_loss'].item(), step)
                self.twriter.add_scalar('Loss/train/kl', loss_dict['kl_loss'].item(), step)

                beta = self.kl_annealing.get_beta()

                if self.args.scheduled_sampling:
                    tf_str = "(per-frame)"
                elif adapt_TeacherForcing:
                    tf_str = "(per-epoch)"
                else:
                    tf_str = "(none)"
                self.tqdm_bar(f'train {tf_str} [tfr: {self.tfr:.1f}], beta: {beta:.3f}', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            # Tensorboard logging (epoch)
            self.twriter.add_scalar('Metrics/train/beta', beta, self.current_epoch)
            self.twriter.add_scalar('Metrics/train/tfr', self.tfr, self.current_epoch)
            self.twriter.add_scalar('Metrics/train/lr', self.scheduler.get_last_lr()[0], self.current_epoch)

            # Keep training-time validation fast with single-sample rollout.
            mse, psnr = self.eval(val_loader, mc_samples=1)

            if psnr > best_psnr:
                best_psnr = psnr
                self.save(os.path.join(self.args.save_root, args.exp_name, f"best.ckpt"))

            if self.current_epoch % self.args.per_save == 0 or self.current_epoch == self.args.num_epoch - 1:
                self.save(os.path.join(self.args.save_root, args.exp_name, f"epoch={self.current_epoch}.ckpt"))
                
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

        print(f"Best PSNR: {best_psnr}")
        
        self.twriter.close()

    @torch.no_grad()
    def eval(self, val_loader=None, mc_samples=None):
        if self.args.fast_train:
            return 0, 0

        if mc_samples is None:
            mc_samples = self.args.mc_samples
        
        if val_loader is None:
            val_loader = self.val_dataloader()

        mse_list = []
        psnr_list = []
        psnr_per_frame_sum = None
        psnr_per_frame_count = 0

        for i, (img, label) in enumerate(pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            mse, psnr, pred = self.val_one_step(img, label, mc_samples=mc_samples)
            mse_list.append(mse.item())
            psnr_list.append(psnr.item())

            # Aggregate frame-wise PSNR across validation batches.
            batch_psnr_per_frame = Generate_PSNR_per_frame(pred, img)
            if psnr_per_frame_sum is None:
                psnr_per_frame_sum = torch.zeros_like(batch_psnr_per_frame)
            psnr_per_frame_sum += batch_psnr_per_frame
            psnr_per_frame_count += 1

            self.tqdm_bar('val', pbar, mse.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            # Visualization: log the first video of the validation set every per_save epochs
            if i == 0 and self.current_epoch % self.args.per_save == 0:
                # TensorBoard add_video expects (N, T, C, H, W)
                # pred: (1, T, 3, H, W)
                self.twriter.add_video('Images/val/prediction', pred, self.current_epoch, fps=25)
                self.twriter.add_video('Images/val/ground_truth', img, self.current_epoch, fps=25)

        avg_mse = sum(mse_list) / len(mse_list)
        avg_psnr = sum(psnr_list) / len(psnr_list)

        print(f"PSNR: {avg_psnr}")
        
        self.twriter.add_scalar('Loss/val/mse', avg_mse, self.current_epoch)
        self.twriter.add_scalar('Metrics/val/psnr', avg_psnr, self.current_epoch)
        if psnr_per_frame_sum is not None and psnr_per_frame_count > 0:
            avg_psnr_per_frame = psnr_per_frame_sum / psnr_per_frame_count

            timesteps = np.arange(1, avg_psnr_per_frame.shape[0])
            psnr_values = avg_psnr_per_frame.detach().cpu().numpy()[1:]  # Skip first frame
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(timesteps, psnr_values, marker='o', linewidth=2)
            ax.set_xlabel('timestep')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('Validation Per-frame PSNR (excluding first frame)')
            ax.grid(True, linestyle='--', alpha=0.3)
            self.twriter.add_figure('Metrics/val/psnr_per_frame_curve', fig, self.current_epoch)
            plt.close(fig)

        return avg_mse, avg_psnr
    
    def training_one_step(self, img, label, adapt_TeacherForcing):  # TODO
        """
        img: (B, T, 3, H, W)
        label: (B, T, 3, H, W)
        """
        B, T, C, H, W = img.shape
        
        img_reshape = img.reshape(-1, C, H, W)
        label_reshape = label.reshape(-1, C, H, W)
        
        imgs_enc_all = self.frame_transformation(img_reshape).reshape(B, T, -1, H, W)
        labels_enc_all = self.label_transformation(label_reshape).reshape(B, T, -1, H, W)
        
        mus = []
        logvars = []
        imgs_pred = []
        
        for t in range(T):
            z, mu, logvar = self.Gaussian_Predictor(imgs_enc_all[:, t], labels_enc_all[:, t])
            mus.append(mu)
            logvars.append(logvar)

            if t == 0:
                curr_frame_enc = imgs_enc_all[:, 0]
            else:
                if self.args.scheduled_sampling:
                    use_tf = True if random.random() < self.tfr else False
                elif adapt_TeacherForcing:
                    use_tf = True
                else:
                    use_tf = False
                
                if use_tf:
                    prev_img = img[:, t-1]  # Use GT frame
                else:
                    prev_img = imgs_pred[-1].detach()  # Use predicted frame
                curr_frame_enc = self.frame_transformation(prev_img)
            
            latent_fusion = self.Decoder_Fusion(curr_frame_enc, labels_enc_all[:, t], z)
            imgs_pred.append(torch.sigmoid(self.Generator(latent_fusion)))
        
        imgs_pred = torch.stack(imgs_pred, dim=1)
        mu_stack = torch.stack(mus, dim=1)
        logvar_stack = torch.stack(logvars, dim=1)

        mse_loss = self.reconstruction_loss(imgs_pred, img)
        beta = self.kl_annealing.get_beta()
        kl_loss = kl_criterion(mu_stack, logvar_stack)

        loss = mse_loss + beta * kl_loss

        # Early stopping when training failed
        if not (torch.isfinite(mse_loss) and torch.isfinite(kl_loss) and torch.isfinite(loss)):
            raise ValueError(
                f"\nWarning: non-finite loss at Epoch {self.current_epoch}. "
                f"mse={mse_loss.item()}, kl={kl_loss.item()}, beta={beta:.6f}. Stop."
            )
        
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()

        return {
            'total_loss': loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss
        }
    
    def val_one_step(self, img, label, mc_samples=1):  # TODO
        """
        img: (B, T, 3, H, W)
        label: (B, T, 3, H, W)
        """
        B, T, C, H, W = img.shape
        
        label_reshape = label.reshape(-1, C, H, W)
        labels_enc_all = self.label_transformation(label_reshape).reshape(B, T, -1, H, W)
        
        imgs_pred = [img[:, 0]]
        for t in range(1, T):
            prev_img = imgs_pred[-1].to(self.args.device)
            curr_frame_enc = self.frame_transformation(prev_img)

            # Monte Carlo inference: sample multiple z and average predictions.
            mc_pred_sum = torch.zeros_like(prev_img)
            for _ in range(mc_samples):
                z = torch.randn(B, self.args.N_dim, H, W, device=self.args.device)
                latent_fusion = self.Decoder_Fusion(curr_frame_enc, labels_enc_all[:, t], z)
                mc_pred_sum += torch.sigmoid(self.Generator(latent_fusion))

            imgs_pred.append(mc_pred_sum / mc_samples)
        
        imgs_pred = torch.stack(imgs_pred, dim=1)
        
        mse_loss = self.mse_criterion(imgs_pred, img)
        psnr = Generate_PSNR(imgs_pred, img)
        return mse_loss, psnr, imgs_pred
        
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                partial=self.args.fast_partial if self.args.fast_train else self.args.partial,
                                stride=self.train_vi_stride)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=self.args.train_shuffle)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.6f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "model_name": self.args.exp_name,
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            if self.args.optim == "Adam":
                self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            else:
                self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
            # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.args.num_epoch)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        
        # Check for NaN/Inf in gradients
        valid_gradients = True
        for param in self.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    valid_gradients = False
                    break
        
        if valid_gradients:
            self.optim.step()
        else:
            print(f"Warning: NaN/Inf detected in gradients at Epoch {self.current_epoch}, skipping step.")


def main(args):
    set_seed(args.seed)

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()

    print("========== Settings ==========")
    print(f"Using loss function: {args.loss_type}")
    print(f"Tail-weighted recon loss: scale={args.tail_weight_scale}, power={args.tail_weight_power}")
    print(f"Using scheduled sampling: {args.scheduled_sampling}")
    print(f"Using train video length: {args.train_vi_len}")
    print(f"Using train stride: {args.train_vi_stride}")
    print(f"Initial LR: {model.args.lr}")
    print(f"Optimizer: {model.optim}")
    print(f"Scheduler: {model.scheduler.__class__.__name__}")
    print(f"kl_anneal_type: {args.kl_anneal_type}, kl_anneal_cycle: {args.kl_anneal_cycle}, kl_anneal_ratio: {args.kl_anneal_ratio}")
    print(f"Teacher Forcing Ratio: {model.tfr}, Teacher Forcing Decay Step: {model.tfr_d_step}, Teacher Forcing Start Decay Epoch: {model.tfr_sde}")
    print("============================\n")
    
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--num_workers',   type=int, default=16)
    parser.add_argument('--exp_name',      type=str, default="",     help="Name of the experiment")
    parser.add_argument('--seed',          type=int, default=42,     help="Random seed for reproducibility")

    # Path
    parser.add_argument('--DR',            type=str, default="data", help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default="ckpt", help="The path to save your data")
    parser.add_argument('--log_root',      type=str, default="runs", help="The root path to save tensorboard logs")
    
    # Data
    parser.add_argument('--batch_size',    type=int,    default=8)
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--train_vi_stride', type=int, default=16,     help="Sliding-window stride for training samples")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--train_shuffle', action='store_true',      help="Whether to shuffle the training dataset")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--lr',            type=float,  default=1e-3,     help="initial learning rate")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=5,      help="Save checkpoint every seted epoch")
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,   help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=25,    help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,   help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--scheduled_sampling', action='store_true',  help="Whether to use scheduled sampling (use predicted frame as input)")
    
    # Evaluation and testing
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--ckpt_path',     type=str,    default=None, help="The path of your checkpoints")
    parser.add_argument('--logdir',        type=str,    default=None, help="The path of your tensorboard logs (for testing)")  
    parser.add_argument('--mc_samples',    type=int,    default=1,    help="MC samples for standalone eval/test mode")
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # KL annealing strategy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='cyc', choices=['cyc', 'mono', 'fixed'],     help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=4,                                           help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,                                       help="")
    parser.add_argument('--kl_beta_min',        type=float, default=1e-2,                                      help="minimum KL beta for numerical stability")
    
    # Loss function arguments
    parser.add_argument('--loss_type',          type=str, default='mse', choices=['l1', 'mse', 'focal'],             help="")
    parser.add_argument('--tail_weight_scale',  type=float, default=0.0, help="Extra weight for later frames in reconstruction loss")
    parser.add_argument('--tail_weight_power',  type=float, default=1.0, help="Power for timestep weighting curve")
    
    args = parser.parse_args()
    if args.train_vi_stride <= 0:
        raise ValueError("--train_vi_stride must be > 0")
    if args.tail_weight_power <= 0:
        raise ValueError("--tail_weight_power must be > 0")
    
    main(args)
