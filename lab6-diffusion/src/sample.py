import os
import time
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
            return self.model(net_input, t).sample # (b, c, w, h)
        else:
            # encoder_hidden_states: (b, seq_len=1, feat_dim=class_emb_size)
            return self.model(x, t, encoder_hidden_states=class_cond.unsqueeze(1)).sample


def sample(args, dataloader, save_subdir):
    checkpoint = torch.load(args.ckpt_path)
    model = ClassConditionedUnet(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with accuracy: {checkpoint['acc']:.4f} from step: {checkpoint['step']} (epoch: {checkpoint['epoch']})")

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

    evaluator = evaluation_model()

    model.eval()
    all_acc = 0
    count = 0
    all_final_images = []
    start_time = time.time()
    
    total_steps = len(scheduler.timesteps)
    capture_indices = np.linspace(0, total_steps - 1, 12, dtype=int)
    denoising_history = []
            
    with torch.no_grad():
        for b_idx, labels in enumerate(dataloader):
            labels = labels.to(device)
            batch_size = labels.shape[0]
                    
            # Sampling
            sample = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
                    
            uncond_labels = torch.zeros_like(labels).to(device)
            for i, t in enumerate(tqdm(scheduler.timesteps, desc=f"Sampling {save_subdir}", leave=False)):
                if args.use_cfg:
                    batched_sample = torch.cat([sample, sample], dim=0)
                    batched_t = torch.full((2 * batch_size,), t.item(), device=device, dtype=torch.long) if t.ndim == 0 else torch.cat([t, t], dim=0).to(device)
                    batched_labels = torch.cat([labels, uncond_labels], dim=0)
                    
                    model_output = model(batched_sample, batched_t, batched_labels)
                    cond_res, uncond_res = torch.chunk(model_output, 2, dim=0)
                    residual = uncond_res + args.cfg_weight * (cond_res - uncond_res)
                else:
                    residual = model(sample, t, labels)
                
                sample = scheduler.step(residual, t, sample).prev_sample
                
                if b_idx == 0 and i in capture_indices:
                    denoising_history.append(sample[:5].detach().cpu())

            # Accuracy
            acc = evaluator.eval(sample, labels)
            all_acc += acc * batch_size
            count += batch_size

            # Save images
            dir = f"{args.save_dir}/{save_subdir}"
            os.makedirs(dir, exist_ok=True)
            for i in range(batch_size):
                img = sample[i]
                img = (img / 2 + 0.5).clamp(0, 1)
                torchvision.utils.save_image(img, f"{dir}/{b_idx * batch_size + i}.png")
            
            if len(all_final_images) == 0:
                all_final_images.append(sample.detach().cpu())

        # Sace result grid
        grid_imgs = torch.cat(all_final_images, dim=0)[:32]
        grid = torchvision.utils.make_grid(grid_imgs, nrow=8, normalize=True, value_range=(-1, 1))
        grid_path = f"{args.save_dir}/{save_subdir}_grid.png"
        torchvision.utils.save_image(grid, grid_path)
        print(f"Saved grid to {grid_path}")

        # Save denoising process
        if len(denoising_history) > 0:
            for img_idx in range(5):
                steps = [h[img_idx] for h in denoising_history]
                strip = torchvision.utils.make_grid(torch.stack(steps), nrow=12, normalize=True, value_range=(-1, 1))
                strip_path = f"{args.save_dir}/{save_subdir}_denoising_process_{img_idx}.png"
                torchvision.utils.save_image(strip, strip_path)
            print(f"Saved 5 denoising process strips to {args.save_dir}")

        avg_acc = all_acc / count
        duration = time.time() - start_time
        print(f"[{save_subdir}] Average Accuracy = {avg_acc:.4f} (Time: {duration:.2f}s)")
        return avg_acc, duration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="images")
    parser.add_argument("--exp-name", type=str, default="")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--class_emb_size", type=int, default=256)
    
    parser.add_argument("--use_cfg", type=eval, default=True)
    parser.add_argument("--cfg_weight", type=float, default=3.0)

    parser.add_argument("--cond_type", type=str, default="cross_attn", choices=["concat", "cross_attn"])
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear", choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    parser.add_argument("--num_test_timesteps", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    basename = os.path.basename(args.ckpt_path).split('.')[0]
    parent_dir = args.ckpt_path.split('/')[-2]
    
    if args.exp_name == "":
        args.exp_name = f"{parent_dir}_{basename}_cfg{args.cfg_weight}_s{args.num_test_timesteps}"
    
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    test_dataloader = get_test_dataloader(
        json_file="test.json",
        objects_file="objects.json",
        batch_size=32,
        shuffle=False,
    )
    new_test_dataloader = get_test_dataloader(
        json_file="new_test.json",
        objects_file="objects.json",
        batch_size=32,
        shuffle=False,
    )

    acc_test, dur_test = sample(args, test_dataloader, "test")
    acc_new, dur_new = sample(args, new_test_dataloader, "new_test")

    result_path = os.path.join(args.save_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write(f"Experiment Configuration:\n")
        f.write(f"  CFG Weight: {args.cfg_weight}\n")
        f.write(f"  Steps: {args.num_test_timesteps}\n")
        f.write(f"  Scheduler: {args.scheduler_type}\n")
        f.write(f"  Beta Schedule: {args.beta_schedule}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {acc_test:.4f} (Time: {dur_test:.2f}s)\n")
        f.write(f"New Test Accuracy: {acc_new:.4f} (Time: {dur_new:.2f}s)\n")
        f.write(f"Total Inference Time: {dur_test + dur_new:.2f}s\n")
    
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()
    