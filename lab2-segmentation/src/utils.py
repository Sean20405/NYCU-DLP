import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.transforms.functional as TF

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')

MODEL_CONFIGS = {
    "unet": {
        "image_size": 572,
        "mask_size": 388,
        "tile_pred": True,
        "random_resize_crop": False,
        "model_name": "unet"
    },
    "resnet_unet": {
        "image_size": 512,
        "mask_size": 512,
        "tile_pred": False,
        "random_resize_crop": True,
        "model_name": "resnet_unet"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the Oxford-IIIT Pet Dataset")

    # Path
    parser.add_argument("--root_dir", type=str, default="dataset/oxford-iiit-pet", help="Root directory of the dataset")
    parser.add_argument("--split_dir", type=str, default="dataset/oxford-iiit-pet/split", help="Directory containing the split files")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts", help="Directory to save checkpoints")

    parser.add_argument("--model_name", type=str, default="unet", help="Name of the model")

    # Model
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "resnet_unet"], help="Type of model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=155, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--scheduler", type=str, default="cosrst", choices=["step", "cos", "cosrst"], help="Scheduler to use for training")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training")

    # Interval
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=20, help="Save interval (epochs)")

    # Inference
    parser.add_argument("--ckpt_path", type=str, default="ckpts/unet_best.pth", help="Path to the checkpoint")
    parser.add_argument("--tile_pred", type=str2bool, nargs='?', const=True, default=None, help="Enable tile prediction")
    parser.add_argument("--fg_threshold", type=float, default=0.5, help="Foreground probability threshold")

    # Train
    parser.add_argument("--random_resize_crop", type=str2bool, nargs='?', const=True, default=None, help="Enable random resize crop")

    args = parser.parse_args()
    
    # Merge with model-specific defaults
    if args.model_type in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_type]
        # Set image_size and mask_size
        args.image_size = config["image_size"]
        args.mask_size = config["mask_size"]
        args.model_name = config["model_name"]

        # Override if None
        if args.tile_pred is None:
            args.tile_pred = config["tile_pred"]
        if args.random_resize_crop is None:
            args.random_resize_crop = config["random_resize_crop"]
    else:
        # Fallback defaults if model_type is unknown (though choices are restricted)
        args.image_size = 512
        args.mask_size = 512
        args.model_name = "unet"

        if args.tile_pred is None:
            args.tile_pred = False
        if args.random_resize_crop is None:
            args.random_resize_crop = False
            
    return args


def dice_score(pred, target):
    epsilon = 1e-7
    pred_mask = (torch.argmax(pred, dim=1) == 0).float()
    target_mask = (target == 0).float()
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        total_dice = 0.0
        for c in range(num_classes):
            probs_c = probs[:, c, :, :]
            targets_c = (targets == c).float()
            intersection = (probs_c * targets_c).sum()
            denominator = probs_c.sum() + targets_c.sum()
            total_dice += (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        return 1. - total_dice / num_classes


def rle_encode(mask):
    """
    Encode binary mask to Run-Length Encoding (RLE) string.
    
    Args:
        mask (numpy.ndarray): Binary mask (1 for foreground, 0 for background).
        
    Returns:
        str: RLE encoded string in column-major order.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_str, shape):
    """
    Decode RLE string to binary mask.
    
    Args:
        rle_str (str): RLE encoded string.
        shape (tuple): Shape of the output mask (width, height).
        
    Returns:
        numpy.ndarray: Binary mask.
    """
    if not rle_str:
        return np.zeros(shape, dtype=np.uint8)
        
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')


    return prediction
