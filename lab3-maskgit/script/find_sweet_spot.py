import subprocess
import re
import os
import argparse

def run_command(cmd):
    print(f"Running: {cmd}")
    # Use Popen to stream output in real-time
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
        output_lines.append(line)
    
    process.stdout.close()
    return_code = process.wait()
    
    stdout = "".join(output_lines)
    if return_code != 0:
        print(f"\nCommand failed with return code {return_code}")
    
    return stdout, "" # stderr is redirected to stdout here

def main():
    parser = argparse.ArgumentParser(description="Find the best sweet-spot for inpainting FID.")
    parser.add_argument('--ckpt', type=str, default='transformer_checkpoints/best_val_loss.pt', help='Path to checkpoint')
    parser.add_argument('--total-iter', type=int, default=24, help='Total iterations in MaskGIT')
    parser.add_argument('--gt-csv', type=str, default='faster-pytorch-fid/test_gt.csv', help='Path to GT CSV')
    parser.add_argument('--partial', type=float, default=1.0, help='Partial dataset to use (1.0 for full FID)')
    parser.add_argument('--mask-func', type=str, default='cosine', help='Mask function to use')
    
    args = parser.parse_args()

    best_fid = float('inf')
    best_spot = -1
    results = []

    # 1. Run inpainting ONCE for all steps
    print(f"\n{'='*40}")
    print(f" Generating inpainting results for all steps...")
    print(f"{'='*40}")
    inpaint_cmd = f"python inpainting.py --load-transformer-ckpt-path {args.ckpt} --total-iter {args.total_iter} --partial {args.partial} --mask-func {args.mask_func}"
    run_command(inpaint_cmd)

    # 2. Iterate through possible sweet spots and calculate FID
    for spot in range(1, args.total_iter + 1):
        print(f"\n{'='*20}")
        print(f" Testing sweet-spot: {spot} / {args.total_iter}")
        print(f"{'='*20}")
        
        # Run FID score calculation using step-specific subfolders
        fid_cmd = f"python faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results_all/{args.mask_func}/step_{spot} --gtcsv-path {args.gt_csv}"
        stdout, stderr = run_command(fid_cmd)
        
        # 3. Parse FID from output
        # Expecting output like "FID:  12.345"
        match = re.search(r"FID:\s*([\d.]+)", stdout)
        if match:
            fid = float(match.group(1))
            print(f">>> Result: sweet-spot {spot} -> FID: {fid}")
            results.append((spot, fid))
            if fid < best_fid:
                best_fid = fid
                best_spot = spot
        else:
            print(f"Warning: Could not parse FID score for sweet-spot {spot}")
            if stderr:
                print("Error output:", stderr)

    # Final summary
    print("\n" + "#"*40)
    print("      SWEET SPOT SEARCH SUMMARY")
    print("#"*40)
    print(f"{'Sweet Spot':<12} | {'FID Score':<10}")
    print("-" * 25)
    for spot, fid in results:
        print(f"{spot:<12} | {fid:<10.4f}")
    
    print("-" * 25)
    if best_spot != -1:
        print(f"BEST SWEET SPOT: {best_spot}")
        print(f"MINIMUM FID:    {best_fid:.4f}")
    else:
        print("No valid FID scores were found.")

if __name__ == "__main__":
    main()
