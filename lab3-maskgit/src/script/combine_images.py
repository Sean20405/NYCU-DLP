import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

def combine_images(image_paths, labels, output_path, mode='v'):
    # Load images
    valid_data = []
    for p, l in zip(image_paths, labels):
        if os.path.exists(p):
            try:
                img = mpimg.imread(p)
                valid_data.append((img, l))
            except Exception as e:
                print(f"Error loading {p}: {e}")
        else:
            print(f"Warning: File {p} not found.")
    
    if not valid_data:
        print("No images found to combine.")
        return

    num_images = len(valid_data)
    
    # Create figure
    if mode == 'v':
        fig, axes = plt.subplots(num_images, 1, figsize=(12, 3 * num_images))
    else:  # horizontal
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for ax, (img, label) in zip(axes, valid_data):
        ax.imshow(img)
        
        if mode == 'v':
            # Label on the left
            ax.set_ylabel(label, rotation=90, labelpad=16, fontsize=16, 
                          verticalalignment='center', horizontalalignment='center')

        else:
            # Label at the bottom
            ax.set_xlabel(label, labelpad=6, fontsize=16)
        
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Adjust layout
    if mode == 'v':
        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05, hspace=0.1)
    else:
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.2, wspace=0.1)
    
    # Save the result
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Successfully saved combined image (mode={mode}) to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images with labels using Matplotlib.")
    parser.add_argument("--indices", type=int, nargs="+", default=[0], help="Indices of test images to combine")
    parser.add_argument("--funcs", type=str, nargs="+", default=["linear", "cosine", "square"], help="Mask functions to compare")
    parser.add_argument("--dir", type=str, default="imga", help="Directory containing the images")
    parser.add_argument("--mode", type=str, choices=['v', 'h'], default='v', help="Orientation: v (vertical) or h (horizontal)")
    
    args = parser.parse_args()
    
    image_paths = []
    labels = []
    
    for f in args.funcs:
        for i in args.indices:
            path = f"{args.dir}/{f}/test_{i}.png"
            image_paths.append(path)
            labels.append(f)

    output_path = f"{args.dir}/func_comparison.png"
            
    combine_images(image_paths, labels, output_path, mode=args.mode)
