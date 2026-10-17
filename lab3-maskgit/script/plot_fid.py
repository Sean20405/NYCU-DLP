import matplotlib.pyplot as plt
import re
import os

def parse_fid(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
        
    data = {}
    current_scheduler = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a scheduler name
        if line.lower() in ['linear', 'cosine', 'square']:
            current_scheduler = line.lower()
            data[current_scheduler] = {'sweet_spots': [], 'fid_scores': []}
            continue
            
        # Check if it's a data line (e.g., "1            | 33.8595")
        # Matches "digits | decimal" or "digits | digits"
        match = re.match(r'^(\d+)\s*\|\s*([\d\.]+)$', line)
        if match and current_scheduler:
            ss = int(match.group(1))
            fid = float(match.group(2))
            data[current_scheduler]['sweet_spots'].append(ss)
            data[current_scheduler]['fid_scores'].append(fid)
            
    return data

def plot_data(data, output_file='fid_plot.png'):
    if not data:
        return

    # Use a modern style
    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Custom color palette
    colors = {
        'linear': '#3498db',  # Blue
        'cosine': '#e74c3c',  # Red
        'square': '#2ecc71'   # Green
    }
    markers = {
        'linear': 'o', 
        'cosine': 's', 
        'square': '^'
    }
    
    all_fids = []
    
    for scheduler, values in data.items():
        if not values['sweet_spots']:
            continue
            
        ax.plot(values['sweet_spots'], values['fid_scores'], 
                 label=scheduler.capitalize(), 
                 color=colors.get(scheduler, '#333333'), 
                 marker=markers.get(scheduler, 'd'),
                 linewidth=2.5,
                 markersize=7,
                 alpha=0.9,
                 markerfacecolor='white',
                 markeredgewidth=2)
        
        all_fids.extend(values['fid_scores'])
        
        # Find minimum for each scheduler and annotate it
        min_fid = min(values['fid_scores'])
        min_idx = values['fid_scores'].index(min_fid)
        min_ss = values['sweet_spots'][min_idx]
        
        ax.annotate(f'Min: {min_fid:.2f}', 
                    xy=(min_ss, min_fid), 
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=9,
                    fontweight='bold',
                    color=colors.get(scheduler, '#333333'),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=colors.get(scheduler), alpha=0.8))

    # Styling the chart
    ax.set_title('Fréchet Inception Distance (FID) vs Sweet Spot', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Sweet Spot (T)', fontsize=13, fontweight='semibold')
    ax.set_ylabel('FID Score (Lower is better)', fontsize=13, fontweight='semibold')
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(frameon=True, fontsize=11, loc='upper right', shadow=True)
    
    # Set y-axis limits to provide some breathing room
    if all_fids:
        min_y = min(all_fids)
        max_y = max(all_fids)
        padding = (max_y - min_y) * 0.15
        ax.set_ylim(min_y - padding, max_y + padding)
    
    # Tick formatting
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"FID plot saved as: {output_file}")

if __name__ == "__main__":
    input_file = 'fid.txt'
    output_image = 'fid_comparison.png'
    
    fid_data = parse_fid(input_file)
    if fid_data:
        plot_data(fid_data, output_image)
    else:
        print("Failed to parse data from fid.txt")
