import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_comparison():
    # Folder paths
    # Before is image_00x.png in lab3_dataset/masked_image
    # After is image_00x.png in test_results/cosine
    
    before_dir = "lab3_dataset/masked_image"
    after_dir = "test_results/cosine"
    output_path = "inpaint_comparison.png"
    
    num_sets = 12
    sets_per_row = 6
    num_blocks = num_sets // sets_per_row
    
    types = ["Before", "After"]
    dirs = [before_dir, after_dir]
    
    # 2 groups high, each group has 2 images (Before/After) -> 4 rows total
    # 6 sets wide -> 6 columns
    fig, axes = plt.subplots(num_blocks * 2, sets_per_row, figsize=(3 * sets_per_row, 3 * num_blocks * 2))
    
    for set_idx in range(num_sets):
        block_idx = set_idx // sets_per_row
        col_idx = set_idx % sets_per_row
        
        # Short filename for each set
        fn_short = f"image_{set_idx:03d}.png"
        
        for type_idx, (label, base_dir) in enumerate(zip(types, dirs)):
            # Row index calculation: block_idx*2 + type_idx
            row_idx = block_idx * 2 + type_idx
            path = os.path.join(base_dir, fn_short)
            ax = axes[row_idx, col_idx]
            
            if os.path.exists(path):
                from PIL import Image
                try:
                    img = Image.open(path)
                    ax.imshow(img)
                    
                    # Row titles on the far left for each block
                    if col_idx == 0:
                        ax.set_ylabel(label, fontsize=24, labelpad=12)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center')
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison image successfully saved to: {output_path}")


if __name__ == "__main__":
    create_comparison()
