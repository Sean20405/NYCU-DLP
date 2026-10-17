import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_cfg_comparison():
    cfg_weights = ["1.0", "2.0", "3.0", "5.0", "8.0"]
    base_dir = "images"
    exp_suffix = "_s50"
    base_exp = "ddim_cross_attn_best_cfg"
    
    # Last 8 images: 24 to 31
    image_indices = list(range(24, 32))
    
    num_rows = len(cfg_weights)
    
    # Pre-concatenate images for each row to ensure zero gap
    row_images = []
    for cfg in cfg_weights:
        cfg_dir = os.path.join(base_dir, f"{base_exp}{cfg}{exp_suffix}", "test")
        imgs = []
        for idx in image_indices:
            img_path = os.path.join(cfg_dir, f"{idx}.png")
            if os.path.exists(img_path):
                imgs.append(np.array(Image.open(img_path).convert('RGB')))
            else:
                # Placeholder if missing
                imgs.append(np.zeros((64, 64, 3), dtype=np.uint8))
        
        # Concatenate horizontally
        row_img = np.concatenate(imgs, axis=1)
        row_images.append(row_img)

    fig, axes = plt.subplots(num_rows, 1, figsize=(16, num_rows * 2))
    plt.subplots_adjust(hspace=0.05) # Small vertical gap between different CFGs
    
    conditions = [
        "blue sphere\nyellow cylinder",
        "red sphere\ncyan cylinder\ncyan cube",
        "gray sphere\npurple cylinder\nblue cube",
        "yellow cube\nbrown cylinder\npurple cube",
        "cyan cube\ngreen cylinder\nblue cube",
        "brown cube\nblue cylinder\nblue sphere",
        "green sphere\nred cylinder\nbrown sphere",
        "blue cylinder\ngray cylinder\ncyan sphere"
    ]

    for r, cfg in enumerate(cfg_weights):
        ax = axes[r]
        ax.imshow(row_images[r])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add CFG label
        ax.set_ylabel(f"CFG = {cfg}", fontsize=14, rotation=0, labelpad=42, va='center')
        
        # Add conditions on top of the first row
        if r == 0:
            for c, cond in enumerate(conditions):
                # Calculate x position (center of each 64-pixel block)
                x_pos = (c * 64) + 32
                ax.text(x_pos, -5, cond, ha='center', va='bottom', fontsize=10)

        # Remove spine
        for spine in ax.spines.values():
            spine.set_visible(False)

    output_path = "cfg_comparison_matplotlib.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Pretty comparison plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_cfg_comparison()
