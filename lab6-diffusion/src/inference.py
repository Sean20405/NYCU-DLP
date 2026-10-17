import torch
import json
import argparse
import os
from torchvision.utils import save_image
from diffusers import DDIMScheduler, DDPMScheduler
from main import ClassConditionedUnet

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load objects mapping
    with open(args.objects_path, 'r') as f:
        objects_map = json.load(f)
    
    num_classes = len(objects_map)
    
    # Create multi-hot label
    labels_tensor = torch.zeros(1, num_classes).to(device)
    for label in args.labels:
        if label in objects_map:
            labels_tensor[0, objects_map[label]] = 1
        else:
            print(f"Warning: Label '{label}' not found in objects.json")

    # Mock args for model initialization
    class MockArgs:
        def __init__(self):
            self.cond_type = args.cond_type
            self.num_classes = num_classes
            self.class_emb_size = args.class_emb_size
            self.image_size = args.image_size

    model_args = MockArgs()
    model = ClassConditionedUnet(model_args).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Setup scheduler
    if args.scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule=args.beta_schedule
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=args.beta_schedule
        )
    
    scheduler.set_timesteps(args.num_test_timesteps)

    # Sampling process
    denoising_images = []
    # indices to save (8 images including start and end)
    save_indices = torch.linspace(0, len(scheduler.timesteps) - 1, 7).long().tolist()

    with torch.no_grad():
        # Start from pure noise (x_T)
        sample = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        denoising_images.append(sample.clone()) # Add the initial noise (Start)
        
        # Unconditional labels for CFG
        uncond_labels = torch.zeros_like(labels_tensor).to(device)

        from tqdm.auto import tqdm
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
            if args.use_cfg:
                # CFG: predict noise for both conditional and unconditional
                batched_sample = torch.cat([sample, sample], dim=0)
                batched_t = torch.full((2,), t.item(), device=device, dtype=torch.long)
                batched_labels = torch.cat([labels_tensor, uncond_labels], dim=0)
                
                model_output = model(batched_sample, batched_t, batched_labels)
                cond_res, uncond_res = torch.chunk(model_output, 2, dim=0)
                
                # Guidance scale
                residual = uncond_res + args.cfg_weight * (cond_res - uncond_res)
            else:
                residual = model(sample, t, labels_tensor)
            
            # Step scheduler
            sample = scheduler.step(residual, t, sample).prev_sample
            
            if i in save_indices:
                denoising_images.append(sample.clone())

    # Save final result
    final_image = (sample + 1.0) / 2.0
    save_image(final_image, args.output_path)
    print(f"Final image saved to {args.output_path}")

    # Save denoising process
    # Concatenate all collected samples into a single grid
    denoising_process = torch.cat(denoising_images, dim=0) # (8, 3, 64, 64)
    denoising_process = (denoising_process + 1.0) / 2.0
    
    process_output_path = args.output_path.replace(".png", "_process.png")
    save_image(denoising_process, process_output_path, nrow=8)
    print(f"Denoising process saved to {process_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--labels", type=str, nargs='+', default=["red sphere", "cyan cylinder", "cyan cube"], help="List of labels to generate")
    parser.add_argument("--output-path", type=str, default="inference_result.png", help="Path to save output image")
    
    parser.add_argument("--objects-path", type=str, default="objects.json")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--class_emb_size", type=int, default=256)
    parser.add_argument("--cond_type", type=str, default="cross_attn", choices=["concat", "cross_attn"])
    
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--num_test_timesteps", type=int, default=50)
    
    parser.add_argument("--use_cfg", type=bool, default=True)
    parser.add_argument("--cfg_weight", type=float, default=3.0)

    args = parser.parse_args()
    inference(args)
