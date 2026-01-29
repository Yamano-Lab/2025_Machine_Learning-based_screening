# coding: utf-8
import os
import shutil
import random
import argparse
from glob import glob

def split_wt_dataset(dataset_dir, wt_folder_name='WT', test_ratio=0.2):
    """
    WTフォルダ内の画像を Train(基準) と Test(検証) に物理的に分割する。
    """
    wt_path = os.path.join(dataset_dir, wt_folder_name)
    test_path = os.path.join(dataset_dir, f"{wt_folder_name}_test")

    if not os.path.exists(wt_path):
        print(f"[Error] WT folder not found: {wt_path}")
        return

    # すでに分割済みかチェック
    if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
        print(f"[Warning] Target folder '{test_path}' already exists and is not empty.")
        print("Skipping split to prevent data loss or double splitting.")
        return

    # 画像取得
    all_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
    total_files = len(all_files)
    
    if total_files == 0:
        print("[Error] No images found in WT folder.")
        return

    # ランダムシャッフル
    random.seed(42) # 再現性のため固定
    random.shuffle(all_files)

    # 分割数計算
    n_test = int(total_files * test_ratio)
    test_files = all_files[:n_test]
    train_files = all_files[n_test:]

    print(f"Total WT files: {total_files}")
    print(f"  -> Moving {n_test} files ({test_ratio*100}%) to '{os.path.basename(test_path)}' (Test Sample)")
    print(f"  -> Keeping {len(train_files)} files in '{wt_folder_name}' (Baseline/Train)")

    # 移動実行
    os.makedirs(test_path, exist_ok=True)
    
    for src in test_files:
        filename = os.path.basename(src)
        dst = os.path.join(test_path, filename)
        shutil.move(src, dst)

    print("\nSplit complete.")
    print(f"You can now treat '{os.path.basename(test_path)}' as a distinct sample.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split WT dataset into Train/Baseline and Test/Validation sets.")
    parser.add_argument('--input_dir', required=True, help="Path to the aligned dataset root (e.g., ./data/aligned_dataset_v4)")
    parser.add_argument('--ratio', type=float, default=0.2, help="Ratio of data to use for testing (default: 0.2)")
    
    args = parser.parse_args()
    
    split_wt_dataset(args.input_dir, test_ratio=args.ratio)