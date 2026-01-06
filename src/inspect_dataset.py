import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from glob import glob

# 設定: データセットの場所
INPUT_DIR = "./data/aligned_dataset_v3"
OUTPUT_DIR = "./data/inspection_results"

def visualize_and_save(file_path, save_path):
    try:
        # 画像読み込み
        img = tiff.imread(file_path) # (64, 64, 2)
        
        # データチェック
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"WARNING: NaN/Inf detected in {os.path.basename(file_path)}")
            return

        # 表示用にRGB変換 (赤=Ch0, 緑=Ch1, 青=0)
        # float32 (0.0-1.0) なのでそのまま表示できるはずだが、
        # 念のためクリップしておく
        h, w, c = img.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = np.clip(img[..., 0], 0, 1) # Red
        rgb[..., 1] = np.clip(img[..., 1], 0, 1) # Green
        
        # 保存
        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        plt.title(f"{os.path.basename(file_path)}\nMin:{img.min():.2f} Max:{img.max():.2f}")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))[:20] # 最初の20枚だけ確認
    
    print(f"Checking first {len(files)} images...")
    
    for i, file_path in enumerate(files):
        save_name = f"check_{i:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        visualize_and_save(file_path, save_path)
        print(f"Saved inspection image: {save_path}")

    print("\n確認完了: ./data/inspection_results フォルダの中身を確認してください。")
    print("赤と緑の細胞が綺麗に映っていれば、データ作成は成功しています。")

if __name__ == "__main__":
    main()