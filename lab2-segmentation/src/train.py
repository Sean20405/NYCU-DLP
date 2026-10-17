import os
import time

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import albumentations as A

from tqdm import tqdm
from oxford_pet import split_dataset
from utils import parse_args, DiceLoss
from models import unet, resnet34_unet
from eval import evaluation

if __name__ == "__main__":
    args = parse_args()

    # Enable cuDNN benchmark to speed up convolution operations for fixed-size inputs
    torch.backends.cudnn.benchmark = True

    image_size = args.image_size
    mask_size = args.mask_size

    # Transformation
    train_transform = A.Compose([
        (A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), p=1.0) 
         if args.random_resize_crop 
         else A.Resize(height=image_size, width=image_size)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=4
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ])
    val_transform = T.Compose([
        T.Resize(size=(image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_mask_transform = T.Compose([
        T.Resize(size=(image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(mask_size),
    ])
    val_mask_transform = T.Compose([
        T.Resize(size=(image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
    ])

    train_dataset, val_dataset, test_dataset = split_dataset(
        args.root_dir,
        args.model_type,
        train_transform=train_transform,
        val_transform=val_transform,
        train_mask_transform=train_mask_transform,
        val_mask_transform=val_mask_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.model_type == "unet":
        model = unet.UNet(3, 3)
    elif args.model_type == "resnet_unet":
        model = resnet34_unet.ResNet34_UNet(3, 3)
    else:
        raise ValueError("Invalid model type")
    model = model.to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor([3.0, 6.0, 1.0]).to(args.device)
    )  # weight according to dataset class distribution
    criterion_dice = DiceLoss()

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    if args.scheduler == "step":
        main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif args.scheduler == "cos":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    elif args.scheduler == "cosrst":
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    else:
        raise ValueError("Invalid scheduler")

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    strftime = time.strftime('%m%d-%H%M%S', time.localtime())

    ckpt_dir = os.path.join(args.ckpt_dir, args.model_name, strftime)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    writer = SummaryWriter(f"runs/{args.model_name}_{strftime}")
    writer.add_text(
        "Parameters",
        f"""
        Model: {args.model_type}
        Learning rate: {args.lr}
        Epochs: {args.epochs}
        Batch size: {args.batch_size}
        Optimizer: {optimizer.__class__.__name__}
        Loss: {criterion_ce.__class__.__name__}, {criterion_dice.__class__.__name__}
        Scheduler: {scheduler.__class__.__name__}
        Model name: {args.model_name}
        Random resize crop: {args.random_resize_crop}
        """
    )

    print("="*20, "Training Configuration", "="*20)
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss: {criterion_ce.__class__.__name__}, {criterion_dice.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Model name: {args.model_name}")
    print(f"Random resize crop: {args.random_resize_crop}")
    print("="*64)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # Training loop
    best_val_dice = 0.0
    train_time = 0.0
    global_step = 0
    
    pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in pbar:
        epoch_start_time = time.time()

        model.train()
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for images, masks, _, _ in train_pbar:
            images = images.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.3 * loss_ce + 0.7 * loss_dice
            loss.backward()
            optimizer.step()
            global_step += 1
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.log_interval == 0:
                writer.add_scalar(
                    "Loss/Total loss",
                    loss.item(),
                    global_step
                )
                writer.add_scalar(
                    "Loss/Cross entropy loss",
                    loss_ce.item(),
                    global_step
                )
                writer.add_scalar(
                    "Loss/Dice loss",
                    loss_dice.item(),
                    global_step
                )
                writer.add_scalar(
                    "Learning rate",
                    optimizer.param_groups[0]['lr'],
                    global_step
                )
        
        scheduler.step()

        # Validation
        val_loss_ce, val_dice = evaluation(model, val_loader, criterion_ce, args.device, args.model_type)
        writer.add_scalar("Validation loss", val_loss_ce, epoch)
        writer.add_scalar("Validation Dice", val_dice, epoch)

        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        train_time += epoch_time
        
        # Update progress bar
        pbar.set_postfix(val_loss=f"{val_loss_ce:.4f}", val_dice=f"{val_dice:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss_ce,
                "val_dice": val_dice
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss_ce,
                "val_dice": val_dice
            }, os.path.join(ckpt_dir, "best.pth"))

    print(f"Total training time: {train_time:.2f} seconds")