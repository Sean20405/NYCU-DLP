import torch

from utils import dice_score
from inference import infer_image
from tqdm import tqdm


def evaluation(model, val_loader, criterion, device, model_type="unet"):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        eval_pbar = tqdm(val_loader, desc="Evaluation", leave=False)
        for images, masks, _, _ in eval_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Use infer_image to get full-sized prediction with padding
            outputs = infer_image(model, images, model_type=model_type)
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_score(outputs, masks).item()
            eval_pbar.set_postfix(loss=f"{loss.item():.4f}")
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    return val_loss, val_dice
