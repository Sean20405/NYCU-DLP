import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import torchvision.transforms.functional as TF
import csv
from tqdm import tqdm
import math
import torch.nn.functional as F

from oxford_pet import split_dataset, get_default_transform
from utils import parse_args, rle_encode, rle_decode
from models import unet, resnet34_unet
from torchvision import transforms


def calculate_dice(pred, target):
    epsilon = 1e-7
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


def safe_reflect_pad(x, pad_left, pad_right, pad_top, pad_bottom):
    while pad_left > 0:
        p = min(pad_left, x.size(-1) - 1)
        x = torch.nn.functional.pad(x, (p, 0, 0, 0), mode='circular')
        pad_left -= p
    while pad_right > 0:
        p = min(pad_right, x.size(-1) - 1)
        x = torch.nn.functional.pad(x, (0, p, 0, 0), mode='circular')
        pad_right -= p
    while pad_top > 0:
        p = min(pad_top, x.size(-2) - 1)
        x = torch.nn.functional.pad(x, (0, 0, p, 0), mode='circular')
        pad_top -= p
    while pad_bottom > 0:
        p = min(pad_bottom, x.size(-2) - 1)
        x = torch.nn.functional.pad(x, (0, 0, 0, p), mode='circular')
        pad_bottom -= p
    return x


def infer_image(model, image, tile_pred=False, model_type="unet"):
    orig_h, orig_w = image.shape[-2], image.shape[-1]
    
    if tile_pred:
        if model_type == "unet":
            tile_size = 572
            infer_size = 388
        else: # resnet_unet
            tile_size = 512
            infer_size = 512
            
        pad_size = (tile_size - infer_size) // 2
        
        n_tiles_w = math.ceil(orig_w / infer_size)
        n_tiles_h = math.ceil(orig_h / infer_size)
        
        pred_w = n_tiles_w * infer_size
        pred_h = n_tiles_h * infer_size
        
        pad_w_right = pred_w - orig_w
        pad_h_bottom = pred_h - orig_h
        
        padded_image = safe_reflect_pad(
            image, 
            pad_size, pad_size + pad_w_right, 
            pad_size, pad_size + pad_h_bottom
        )
        
        # Determine out_channels from a dummy run
        test_tile = padded_image[:, :, 0:tile_size, 0:tile_size]
        with torch.no_grad():
            test_out = model(test_tile)
        out_channels = test_out.shape[1]
        
        prediction = torch.zeros((1, out_channels, pred_h, pred_w), device=image.device)
        
        for y_idx in range(n_tiles_h):
            for x_idx in range(n_tiles_w):
                y = y_idx * infer_size
                x = x_idx * infer_size
                
                tile = padded_image[:, :, y:y+tile_size, x:x+tile_size]
                out = model(tile)
                prediction[:, :, y:y+infer_size, x:x+infer_size] = out
        
        prediction = prediction[:, :, :orig_h, :orig_w]
    else:
        if model_type == "unet":
            pad_size = 92
            padded_image = safe_reflect_pad(
                image, 
                pad_size, pad_size, 
                pad_size, pad_size
            )
            prediction = model(padded_image)
        else: # resnet_unet
            prediction = model(image)

        if prediction.shape[-2:] != (orig_h, orig_w):
            prediction = TF.resize(
                prediction, 
                [orig_h, orig_w], 
                interpolation=TF.InterpolationMode.BILINEAR
            )
            
    return prediction


def get_tta_prediction(model, image, model_type, tile_pred, scales, orig_size):
    # orig_size is (W, H)
    total_pred = 0
    for scale in scales:
        if scale == "orig":
            s_image = image
        else:
            s_image = TF.resize(image, [scale, scale])
        
        # Forward
        pred = infer_image(model, s_image, tile_pred, model_type)
        
        # Flip
        pred_flip = infer_image(model, TF.hflip(s_image), tile_pred, model_type)
        pred = (pred + TF.hflip(pred_flip)) / 2
        
        # Resize back to original
        if (pred.shape[-2], pred.shape[-1]) != (orig_size[1], orig_size[0]):
            pred = TF.resize(pred, [orig_size[1], orig_size[0]], interpolation=TF.InterpolationMode.BILINEAR)
        
        total_pred += pred
        
    return total_pred / len(scales)


if __name__ == "__main__":
    args = parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    state_dict = ckpt['model_state_dict']
    
    train_dataset, val_dataset, test_dataset = split_dataset(
        args.root_dir, args.model_type, get_default_transform()
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset.transform = test_transform
    test_dataset.mask_transform = transforms.Lambda(lambda x: x)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Must be 1 to handle arbitrary original sizes
        shuffle=False
    )
    
    if args.model_type == "unet":
        model = unet.UNet(3, 3)
        scales = ["orig", 320, 448, 572, 640, 768]  # best for "diceLoss3Class_warmup_diceWeight0.7_CEClassWeight_cosrs"
    elif args.model_type == "resnet_unet":
        model = resnet34_unet.ResNet34_UNet(3, 3)
        scales = [576, 640, 384, 960, 448]  # best for "resnet_warmup_cosrst_randomResizeCrop_correctArch"
    else:
        raise ValueError("Invalid model type")
    model = model.to(args.device)
    
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint from {args.ckpt_path}")
    print(f"Epoch: {ckpt['epoch']}")
    print(f"Validation loss: {ckpt['val_loss']}")
    print(f"Validation Dice: {ckpt['val_dice']}")
    print(f"Tile prediction: {args.tile_pred}")
    
    submission_results = []
    
    with torch.no_grad():
        for i, (images, masks, original_sizes, paths) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(args.device)  # (1, 3, H, W)
            orig_w, orig_h = original_sizes[0][0].item(), original_sizes[1][0].item()

            prediction = get_tta_prediction(
                model, images, args.model_type, 
                args.tile_pred, scales, (orig_w, orig_h)
            )
            
            probs = torch.softmax(prediction, dim=1)
            fg_prob = probs[0, 0, :, :]
            preds = torch.argmax(prediction, dim=1)  # (1, H, W)
            binary_mask = ((preds == 0) & (fg_prob > args.fg_threshold)).byte()
            
            pred_np = binary_mask[0].cpu().numpy().astype(np.uint8)
            
            rle = rle_encode(pred_np)
            image_id = os.path.splitext(os.path.basename(paths[0]))[0]
            submission_results.append((image_id, rle))
    
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'encoded_mask'])
        writer.writerows(submission_results)
    
    print(f"Submission saved to submission.csv")

    print(f"\nEvaluating DICE score from submission.csv...")
    predictions = {row[0]: row[1] for row in submission_results}
    
    total_dice = 0.0
    count = 0
    image_dice_list = []
    
    for i in range(len(test_dataset)):
        img_path = test_dataset.images[i]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        
        if image_id not in predictions:
            continue

        rle_str = predictions[image_id]

        gt_mask_path = test_dataset.masks[i]
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        gt_binary = (gt_mask == 1).astype(np.uint8)
        
        pred_mask = rle_decode(rle_str, gt_binary.shape)
        
        dice = calculate_dice(pred_mask, gt_binary)
        if hasattr(dice, 'item'):
            dice = dice.item()
        total_dice += dice
        count += 1
        
        image_dice_list.append((dice, image_id, img_path, gt_mask_path, pred_mask))

    avg_dice = total_dice / count
    print(f"Average DICE score from submission.csv: {avg_dice:.4f}")

    print("Saving the 10 images with the worst DICE score...")
    image_dice_list.sort(key=lambda x: x[0])
    os.makedirs("results/worst_10", exist_ok=True)
    
    for rank, (dice, image_id, img_path, gt_mask_path, pred_mask) in enumerate(image_dice_list[:10]):
        orig = Image.open(img_path).convert("RGB")
        
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
        gt_binary = (gt_mask == 1).astype(np.uint8)
        gt_img = Image.fromarray(gt_binary * 255).convert("RGB")
        
        pred_img = Image.fromarray(pred_mask * 255).convert("RGB")
        
        orig = orig.resize((256, 256))
        gt_img = gt_img.resize((256, 256))
        pred_img = pred_img.resize((256, 256))
        
        # Create a composite image: original, GT, Pred side-by-side
        comp = Image.new("RGB", (256 * 3, 256))
        comp.paste(orig, (0, 0))
        comp.paste(gt_img, (256, 0))
        comp.paste(pred_img, (256 * 2, 0))
        
        save_path = f"results/worst_10/{rank+1:02d}_{image_id}_dice_{dice:.4f}.jpg"
        comp.save(save_path)

