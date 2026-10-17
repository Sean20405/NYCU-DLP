import os
import argparse
import numpy as np
from tqdm.auto import tqdm
import wandb

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, UNet2DConditionModel

from evaluator import evaluation_model
from dataset import get_dataloader, get_test_dataloader

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class ClassConditionedUnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cond_type = args.cond_type
        # The embedding layer
        self.class_emb = nn.Linear(args.num_classes, args.class_emb_size)

        if self.cond_type == "concat":  # Concat condition embedding with image
            self.model = UNet2DModel(
                sample_size=args.image_size,
                in_channels=3 + args.class_emb_size,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(64, 128, 256, 512), 
                down_block_types=( 
                    "DownBlock2D",        # a regular ResNet downsampling block
                    "DownBlock2D",
                    "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                    "AttnDownBlock2D",
                ), 
                up_block_types=(
                    "AttnUpBlock2D", 
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:  # Cross-Attention conditioning
            self.model = UNet2DConditionModel(
                sample_size=args.image_size,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(64, 128, 256, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=args.class_emb_size,  # Cross attention with embedding
            )
  
    def forward(self, x, t, class_labels):
        b, c, w, h = x.shape

        class_cond = self.class_emb(class_labels.float())  # (b, class_emb_size)
        
        if self.cond_type == "concat":
            class_cond_reshape = class_cond.view(b, class_cond.shape[1], 1, 1).expand(b, class_cond.shape[1], w, h)
            # x: (b, c, w, h), class_cond: (b, class_emb_size, w, h)
            net_input = torch.cat((x, class_cond_reshape), 1)  # (b, c + class_emb_size, w, h)
            return self.model(net_input, t).sample  # (b, c, w, h)
        else:
            # encoder_hidden_states: (b, seq_len=1, feat_dim=class_emb_size)
            return self.model(x, t, encoder_hidden_states=class_cond.unsqueeze(1)).sample


def train(args):
    # Setup training components
    model = ClassConditionedUnet(args).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=args.beta_schedule
    )
    if args.scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=args.beta_schedule
        )
    else:
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule=args.beta_schedule
        )
    scheduler.set_timesteps(args.num_test_timesteps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-6,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    

    train_dataloader = get_dataloader(
        root_dir="iclevr",
        json_file="train.json",
        objects_file="objects.json",
        batch_size=args.batch_size,
        shuffle=True,
    )

    eval_loader = get_test_dataloader(
        json_file="test.json",
        objects_file="objects.json",
        batch_size=32,
        shuffle=False,
        num_workers=0,
        subset=args.eval_subset
    )

    loss_fn = nn.MSELoss()
    total_step = 0
    best_acc = 0.0

    evaluator = evaluation_model()

    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            total_step += 1

            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            if args.use_cfg and torch.rand(1).item() < 0.2:  # Randomly drop class condition for CFG
                train_labels = torch.zeros_like(labels)
            else:
                train_labels = labels

            # Sample noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            predicted_noise = model(noisy_images, timesteps, train_labels)

            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "step": total_step})

        # Eval
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            all_acc = 0
            count = 0
            
            with torch.no_grad():
                for b_idx, labels in enumerate(eval_loader):
                    labels = labels.to(device)
                    batch_size = labels.shape[0]
                    
                    # Sampling
                    sample = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
                    
                    uncond_labels = torch.zeros_like(labels).to(device)
                    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling", leave=False)):
                        if args.use_cfg:
                            # Batch cond and uncond for faster sampling
                            # model input shape: (batch_size * 2, 3 + class_emb_size, image_size, image_size)
                            batched_sample = torch.cat([sample, sample], dim=0)
                            batched_t = torch.full((2 * batch_size,), t.item(), device=device, dtype=torch.long) if t.ndim == 0 else torch.cat([t, t], dim=0).to(device)
                            batched_labels = torch.cat([labels, uncond_labels], dim=0)
                            
                            model_output = model(batched_sample, batched_t, batched_labels)
                            cond_res, uncond_res = torch.chunk(model_output, 2, dim=0)
                            residual = uncond_res + args.cfg_weight * (cond_res - uncond_res)
                        else:
                            residual = model(sample, t, labels)
                        
                        sample = scheduler.step(residual, t, sample).prev_sample

                    # Accuracy
                    acc = evaluator.eval(sample, labels)
                    all_acc += acc * batch_size
                    count += batch_size
                    
                    # Log generated images for the first batch
                    if b_idx == 0:
                        final_images = (sample + 1.0) / 2.0
                        wandb.log({"generated_images": [wandb.Image(img) for img in final_images[:8]], "epoch": epoch})

                avg_acc = all_acc / count

                if avg_acc >= best_acc:
                    best_acc = avg_acc
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'acc': avg_acc,
                        'step': total_step,
                        'epoch': epoch + 1
                    }
                    torch.save(checkpoint, f"{args.save_dir}/best.pth")
                    print(f"New best model saved with accuracy {best_acc:.4f} at step {total_step}")

                if (epoch + 1) % args.save_interval == 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'acc': avg_acc,
                        'step': total_step,
                        'epoch': epoch + 1
                    }
                    torch.save(checkpoint, f"{args.save_dir}/epoch_{epoch+1}.pth")
                    print(f"Model saved at epoch {epoch+1}")

                print(f"Epoch {epoch + 1}: Accuracy = {avg_acc:.4f}")
                wandb.log({"accuracy": avg_acc, "epoch": epoch})

            model.train()

        lr_scheduler.step()
        wandb.log({"lr": lr_scheduler.get_last_lr()[0], "epoch": epoch})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ddpm_unet")
    parser.add_argument("--save-dir", type=str, default=f"ckpt/ddpm_unet")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--class_emb_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    
    parser.add_argument("--use_cfg", type=eval, default=True)
    parser.add_argument("--cfg_weight", type=float, default=3.0)
    
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_subset", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=20)
    
    parser.add_argument("--cond_type", type=str, default="cross_attn", choices=["concat", "cross_attn"])
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear", choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    parser.add_argument("--num_test_timesteps", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    args.save_dir = f"results/{args.exp_name}"
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    wandb.init(project="NYCU-DLP-Lab6", name=args.exp_name, config=args)
    
    # Define custom metrics for different x-axes
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("loss", step_metric="step")
    wandb.define_metric("accuracy", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("generated_images", step_metric="epoch")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train(args)
    wandb.finish()

if __name__ == "__main__":
    main()
    