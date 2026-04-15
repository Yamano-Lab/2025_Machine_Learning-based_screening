# coding: utf-8
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import seaborn as sns
from glob import glob
from tensorflow.keras.models import load_model

def get_average_error_map(folder_path, model, batch_size=32):
    """指定フォルダの平均誤差マップ(64,64,2)を計算して返す"""
    files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
    if not files: return None
    
    accumulated_diff = None
    count = 0
    batch_images = []
    
    print(f"  Calculating error map for: {os.path.basename(folder_path)} ({len(files)} cells)...")
    
    for file_path in files:
        try:
            img = tiff.imread(file_path)
            if img.ndim == 3 and img.shape[-1] == 2:
                batch_images.append(img)
            
            if len(batch_images) >= batch_size:
                batch_arr = np.array(batch_images).astype('float32')
                recon = model.predict(batch_arr, verbose=0)
                diff = np.abs(batch_arr - recon)
                batch_sum = np.sum(diff, axis=0)
                
                if accumulated_diff is None: accumulated_diff = batch_sum
                else: accumulated_diff += batch_sum
                count += len(batch_images)
                batch_images = []
        except: pass

    if batch_images:
        batch_arr = np.array(batch_images).astype('float32')
        recon = model.predict(batch_arr, verbose=0)
        diff = np.abs(batch_arr - recon)
        batch_sum = np.sum(diff, axis=0)
        if accumulated_diff is None: accumulated_diff = batch_sum
        else: accumulated_diff += batch_sum
        count += len(batch_images)
        
    if count == 0: return None
    return accumulated_diff / count

def visualize_differential(input_dir, model_dir, output_dir, wt_folder_name='WT', heatmap_range=None, normalize=False):
    print(f"Loading model from {model_dir}...")
    autoencoder = load_model(os.path.join(model_dir, 'best_autoencoder.keras'), compile=False)
    
    # 1. WTの平均エラーマップを計算（基準）
    wt_path = os.path.join(input_dir, wt_folder_name)
    if not os.path.exists(wt_path):
        print(f"[Error] WT folder not found at {wt_path}")
        return

    print("=== Step 1: Calculating WT Baseline ===")
    wt_error_map = get_average_error_map(wt_path, autoencoder)
    if wt_error_map is None: return

    # WT自体の可視化
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 各変異株との差分を計算
    print("=== Step 2: Calculating Differential Maps (Aggregation Phase) ===")
    
    sample_folders = []
    wt_abs_path = os.path.abspath(wt_path)
    
    for root, dirs, files in os.walk(input_dir):
        # Skip WT folder
        if os.path.abspath(root) == wt_abs_path:
            continue
            
        # Check for tif files
        has_tif = any(f.endswith('.tif') or f.endswith('.tiff') for f in files)
        if has_tif:
            sample_folders.append(root)
    
    sample_folders.sort()
    
    results = []
    global_max_val = 0.0
    
    for folder in sample_folders:
        # Create a unique name based on relative path (e.g., Candidates_16-1)
        rel_path = os.path.relpath(folder, input_dir)
        sample_name = rel_path.replace(os.path.sep, '_')
        
        mutant_error_map = get_average_error_map(folder, autoencoder)
        if mutant_error_map is None: continue
        
        # 引き算 (Mutant - WT)
        diff_map = mutant_error_map - wt_error_map
        
        if normalize:
            max_diff = np.max(np.abs(diff_map))
            if max_diff > 0:
                diff_map = diff_map / max_diff
                # print(f"  [Info] Normalized diff_map for {sample_name}")
        
        results.append((sample_name, diff_map))
        
        # Update global max
        current_max = np.max(np.abs(diff_map))
        if current_max > global_max_val:
            global_max_val = current_max

    # Determine final scale
    if heatmap_range is not None:
        final_max_val = heatmap_range
        print(f"Using provided fixed range: +/- {final_max_val}")
    elif normalize:
        final_max_val = 1.0
        print(f"Using normalized range: +/- 1.0")
    else:
        final_max_val = global_max_val
        print(f"Using detected global max range: +/- {final_max_val:.6f}")

    print("=== Step 3: Generating Heatmaps (Visualization Phase) ===")
    
    for sample_name, diff_map in results:
        plt.figure(figsize=(15, 6))
        
        # Combined (Red+Green channel average)
        combined_diff = np.mean(diff_map, axis=-1)
        
        plt.subplot(1, 3, 1)
        sns.heatmap(combined_diff, cmap='PuOr_r', center=0, vmin=-final_max_val, vmax=final_max_val)
        plt.title(f"{sample_name} - WT\nCombined Difference" + (" (Normalized)" if normalize else ""))
        plt.axis('off')
        
        # Red Ch (Chloroplast)
        plt.subplot(1, 3, 2)
        diff_r = diff_map[..., 0]
        sns.heatmap(diff_r, cmap='PuOr_r', center=0, vmin=-final_max_val, vmax=final_max_val)
        plt.title("Chloroplast Diff (Red Ch)")
        plt.axis('off')

        # Green Ch (Pyrenoid)
        plt.subplot(1, 3, 3)
        diff_g = diff_map[..., 1]
        sns.heatmap(diff_g, cmap='PuOr_r', center=0, vmin=-final_max_val, vmax=final_max_val)
        plt.title("Pyrenoid Diff (Green Ch)")
        plt.axis('off')

        save_path = os.path.join(output_dir, f"diff_anomaly_{sample_name}_vs_WT.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved differential map: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Aligned dataset folder (v4/v5)")
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output_dir', default='./results/Differential_Anomalies')
    parser.add_argument('--wt_name', default='WT', help="Name of the WT folder")
    parser.add_argument('--heatmap_range', type=float, default=None, help="Fixed range for heatmap (e.g., 0.1)")
    parser.add_argument('--normalize', action='store_true', help="Normalize heatmap values to [-1, 1]")
    args = parser.parse_args()
    
    visualize_differential(args.input_dir, args.model_dir, args.output_dir, args.wt_name, args.heatmap_range, args.normalize)