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

def visualize_differential(input_dir, model_dir, output_dir, wt_folder_name='WT'):
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
    print("=== Step 2: Calculating Differential Maps ===")
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir() and os.path.basename(f.path) != wt_folder_name]
    
    for folder in subfolders:
        sample_name = os.path.basename(folder)
        mutant_error_map = get_average_error_map(folder, autoencoder)
        if mutant_error_map is None: continue
        
        # 引き算 (Mutant - WT)
        # 正の値(Red): WTよりエラーが大きい場所（異常）
        # 負の値(Blue): WTよりエラーが小さい場所
        diff_map = mutant_error_map - wt_error_map
        
        # 可視化
        plt.figure(figsize=(15, 6))
        
        # Combined (Red+Green channel average)
        combined_diff = np.mean(diff_map, axis=-1)
        max_val = np.max(np.abs(combined_diff)) # 0を中心に対称にする
        
        plt.subplot(1, 3, 1)
        # coolwarm: 赤=プラス(異常), 青=マイナス(正常/消失), 白=変化なし
        sns.heatmap(combined_diff, cmap='vlag', center=0, vmin=-max_val, vmax=max_val)
        plt.title(f"{sample_name} - WT\nCombined Difference")
        plt.axis('off')
        
        # Red Ch (Chloroplast)
        plt.subplot(1, 3, 2)
        diff_r = diff_map[..., 0]
        max_r = np.max(np.abs(diff_r))
        sns.heatmap(diff_r, cmap='vlag', center=0, vmin=-max_r, vmax=max_r)
        plt.title("Chloroplast Diff (Red Ch)")
        plt.axis('off')

        # Green Ch (Pyrenoid)
        plt.subplot(1, 3, 3)
        diff_g = diff_map[..., 1]
        max_g = np.max(np.abs(diff_g))
        sns.heatmap(diff_g, cmap='vlag', center=0, vmin=-max_g, vmax=max_g)
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
    args = parser.parse_args()
    
    visualize_differential(args.input_dir, args.model_dir, args.output_dir, args.wt_name)