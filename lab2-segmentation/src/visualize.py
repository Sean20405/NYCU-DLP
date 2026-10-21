"""Save a qualitative result figure for a trained model.

The script predicts every test image with the same pipeline as inference.py,
prints the average Dice score, and plots randomly chosen images with their Dice.
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inference import calculate_dice, get_tta_prediction
from models import resnet34_unet, unet
from oxford_pet import get_default_transform, split_dataset
from utils import str2bool

# Test-time scales tuned for the submitted checkpoints (see inference.py).
SCALES = {"unet": ["orig", 320, 448, 572, 640, 768], "resnet_unet": [576, 640, 384, 960, 448]}


def predict_mask(model, image, model_type, tile_pred, fg_threshold, size):
    prediction = get_tta_prediction(model, image, model_type, tile_pred, SCALES[model_type], size)
    fg_prob = torch.softmax(prediction, dim=1)[0, 0]
    return ((prediction.argmax(dim=1)[0] == 0) & (fg_prob > fg_threshold)).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--model_type", choices=["unet", "resnet_unet"], default="unet")
    parser.add_argument("--root_dir", default="dataset/oxford-iiit-pet")
    parser.add_argument("--tile_pred", type=str2bool, default=False)
    parser.add_argument("--fg_threshold", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    _, _, test_set = split_dataset(args.root_dir, args.model_type, get_default_transform())
    test_set.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_set.mask_transform = transforms.Lambda(lambda x: x)

    model = unet.UNet(3, 3) if args.model_type == "unet" else resnet34_unet.ResNet34_UNet(3, 3)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)["model_state_dict"])
    model = model.to(args.device).eval()

    chosen = set(random.Random(args.seed).sample(range(len(test_set)), args.num_samples))
    dice_all, shown = [], []
    with torch.no_grad():
        for i in range(len(test_set)):
            image, _, size, path = test_set[i]
            pred = predict_mask(model, image.unsqueeze(0).to(args.device), args.model_type,
                                args.tile_pred, args.fg_threshold, size)
            gt = (np.array(Image.open(test_set.masks[i]).convert("L")) == 1).astype(np.uint8)
            dice = float(calculate_dice(pred, gt))
            dice_all.append(dice)
            if i in chosen:
                shown.append((np.array(Image.open(path).convert("RGB")), gt, pred, dice))
    print(f"Average Dice on {len(dice_all)} test images: {np.mean(dice_all):.4f}")
    print("Dice of shown images:", [round(s[3], 4) for s in shown])

    fig, axes = plt.subplots(4, len(shown), figsize=(2 * len(shown), 8.5), squeeze=False)
    for col, (img, gt, pred, dice) in zip(axes.T, shown):
        overlay = img.copy()
        overlay[pred == 1] = (0.5 * overlay[pred == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        for ax, content, cmap in zip(col, [img, gt, pred, overlay], [None, "gray", "gray", None]):
            ax.imshow(content, cmap=cmap)
            ax.axis("off")
        col[0].set_title(f"Dice {dice:.3f}", fontsize=9)
    for ax, label in zip(axes[:, 0], ["Input", "Ground truth", "Prediction", "Overlay"]):
        ax.text(-0.05, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)


if __name__ == "__main__":
    main()
