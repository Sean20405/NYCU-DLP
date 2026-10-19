import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.writer = SummaryWriter(f"runs/{args.model_name}_{time.strftime('%m%d_%H%M%S')}")
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
        os.makedirs(f"transformer_checkpoints/{args.model_name}", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        self.optim.zero_grad()
        for i, images in enumerate(pbar):
            images = images.to(self.args.device)
            logits, targets = self.model(images)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1)
            
            # Gradient Accumulation
            loss = loss / self.args.accum_grad
            loss.backward()
            
            if (i + 1) % self.args.accum_grad == 0 or (i + 1) == len(train_loader):
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), 1.0)
                self.optim.step()
                self.optim.zero_grad()
            
            total_loss += loss.item() * self.args.accum_grad
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item() * self.args.accum_grad:.4f}")

            self.writer.add_scalar('Loss/train', loss.item() * self.args.accum_grad, epoch * len(train_loader) + i)
            self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        
        self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader)
            for i, images in enumerate(pbar):
                images = images.to(self.args.device)
                logits, targets = self.model(images)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                pbar.set_description(f"Eval Epoch: {epoch}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def visualize_inpainting(self, val_loader, epoch):
        self.model.eval()
        image = next(iter(val_loader)).to(self.args.device)
        logits, _ = self.model(image) 
        
        # Predict tokens (exclude mask token from prediction output if possible)
        # logits: (b, 256, 1025)
        z_indices_predict = torch.max(logits[:, :, :self.model.mask_token_id], dim=-1)[1]
        
        # Decode the prediction
        shape = (image.shape[0], 16, 16, 256)
        z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
        z_q = z_q.permute(0, 3, 1, 2)
        decoded_img = self.model.vqgan.decode(z_q)
        
        # Denormalize for visualization
        mean = torch.tensor([0.4816, 0.4324, 0.3845], device=self.args.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2602, 0.2518, 0.2537], device=self.args.device).view(1, 3, 1, 1)
        decoded_img = torch.clamp((decoded_img * std) + mean, 0, 1)
        image_ori = torch.clamp((image * std) + mean, 0, 1)
        
        # Log to TensorBoard
        grid = torch.cat([image_ori, decoded_img], dim=0)
        grid = vutils.make_grid(grid, nrow=image.shape[0], normalize=False)
        self.writer.add_image(f'Validation/OneIter_Reconstruction', grid, epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.transformer.parameters(), 
            lr=self.args.learning_rate, 
            betas=(0.9, 0.96)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args.epochs, 
            eta_min=1e-6
        )
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=20, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--model-name', type=str, default='model', help='Model name.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_val_loss = float('inf')
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch)
        
        # Visualization
        train_transformer.visualize_inpainting(val_loader, epoch)
        
        if epoch % args.save_per_epoch == 0:
            torch.save(
                train_transformer.model.transformer.state_dict(),
                os.path.join("transformer_checkpoints", args.model_name, f"epoch_{epoch}.pt")
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                train_transformer.model.transformer.state_dict(),
                os.path.join("transformer_checkpoints", args.model_name, f"best_val_loss.pt")
            )
    train_transformer.writer.close()
