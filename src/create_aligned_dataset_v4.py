# coding: utf-8
import argparse
import os
import math
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from glob import glob
from skimage.measure import regionprops
from skimage.transform import resize, rotate
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from joblib import Parallel, delayed

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')

def process_single_file(file_path, input_root, output_root, model_name='2D_versatile_fluo'):
    """
    1つの画像ファイルを処理し、厳格な基準で細胞を抽出・整列して保存する。
    ディレクトリ構造を維持して保存する (v3準拠)。
    """
    try:
        # --- 1. 出力パスの決定 (Folder Structure Preservation) ---
        # 入力ルートからの相対パスを取得 (例: "WT/sample1.tif")
        rel_path = os.path.relpath(file_path, input_root)
        # フォルダ部分 (例: "WT")
        rel_dir = os.path.dirname(rel_path)
        # ファイル名 (例: "sample1")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 保存先ディレクトリ (例: output_root/WT)
        save_dir = os.path.join(output_root, rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # --- 2. 画像読み込みとStarDist ---
        # 並列処理時の競合を防ぐため、各ワーカーでモデルロード (多少オーバーヘッドはあるが安全)
        # GPU無効化 (CPU並列処理のため)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            model = StarDist2D.from_pretrained(model_name)
        except:
            # 読み込み済みの場合のフォールバック等は省略、基本は都度ロード
            model = StarDist2D.from_pretrained(model_name)

        image = tiff.imread(file_path)
        
        if image.ndim == 3 and image.shape[-1] >= 2:
            red_channel = image[..., 0]   # Chloroplast
            green_channel = image[..., 1] # Pyrenoid
            seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
        else:
            return []

        normalized_seg = normalize(seg_channel)
        labels, _ = model.predict_instances(normalized_seg)
        props = regionprops(labels)

        p99_red = np.percentile(red_channel, 99.9)
        p99_green = np.percentile(green_channel, 99.9)
        
        saved_files = []
        H, W = red_channel.shape
        count = 0

        for prop in props:
            # --- 厳格なフィルタリング (Strict Filtering) ---
            
            # 1. 見切れ除去 (Border Exclusion) - 5pxマージン
            minr, minc, maxr, maxc = prop.bbox
            if minr <= 5 or minc <= 5 or maxr >= H - 5 or maxc >= W - 5:
                continue

            # 2. 形状フィルタ (Strict Shape)
            if prop.area < 200 or prop.area > 8000: continue
            if prop.eccentricity > 0.95: continue # 細長いゴミを除去
            if prop.solidity < 0.9: continue      # 凹凸のあるものを除去
            
            circularity = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            if circularity < 0.8: continue        # いびつなものを除去

            # --- センタリングと切り出し (Centering & Cropping) ---
            
            # 3. 重心基準のパディング付き切り出し
            cy, cx = map(int, prop.centroid)
            
            # 回転してもはみ出さない十分なサイズ (長辺の1.5倍)
            crop_size_raw = int(max(prop.bbox_height, prop.bbox_width) * 1.5)
            crop_size_raw = max(crop_size_raw, 64) # 最低64
            half_size = crop_size_raw // 2

            # 画像全体をパディング (重心が端にあっても切り出せるように)
            pad_amount = crop_size_raw
            padded_red = np.pad(red_channel, pad_amount, mode='constant')
            padded_green = np.pad(green_channel, pad_amount, mode='constant')
            padded_mask = np.pad(labels == prop.label, pad_amount, mode='constant')
            
            # パディング後の座標
            cy_pad, cx_pad = cy + pad_amount, cx + pad_amount
            
            # 切り出し
            crop_red = padded_red[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]
            crop_green = padded_green[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]
            crop_mask = padded_mask[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]

            # 背景マスク
            crop_red = crop_red * crop_mask
            crop_green = crop_green * crop_mask

            # --- アライメント (Alignment) ---
            
            # クロップ内での重心再計算
            try:
                m = regionprops(crop_mask.astype(int), intensity_image=crop_red)[0]
                ccy, ccx = m.centroid
            except IndexError: continue

            if np.max(crop_green) < p99_green * 0.1: continue

            try:
                mg = regionprops(crop_mask.astype(int), intensity_image=crop_green)[0]
                pcy, pcx = mg.weighted_centroid
            except IndexError: continue
            
            dy, dx = pcy - ccy, pcx - ccx
            
            if dy**2 + dx**2 < 2.0:
                angle = 0
            else:
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                angle = angle_deg - 90 # ピレノイドを下に

            rotated_red = rotate(crop_red, angle, resize=False, preserve_range=True)
            rotated_green = rotate(crop_green, angle, resize=False, preserve_range=True)
            
            # --- 最終リサイズ (64x64) ---
            final_size = 64
            center_y, center_x = rotated_red.shape[0] // 2, rotated_red.shape[1] // 2
            y1 = center_y - final_size // 2
            x1 = center_x - final_size // 2
            
            # 中心から64x64を切り出し (またはリサイズ)
            # ここでは解像度を統一するため、resizeを使用
            final_red = resize(rotated_red, (final_size, final_size), anti_aliasing=True)
            final_green = resize(rotated_green, (final_size, final_size), anti_aliasing=True)
            
            # 正規化
            final_red = np.clip(final_red / p99_red, 0, 1)
            final_green = np.clip(final_green / p99_green, 0, 1)

            # 保存
            save_img = np.stack([final_red, final_green], axis=-1).astype(np.float32)
            
            # v3準拠のファイル名: 元ファイル名_cell_番号.tif
            # フラットにはせず、フォルダ内に入れるのでファイル名はシンプルでも良いが、一意性を保つため付与
            out_filename = f"{base_name}_cell_{count:04d}.tif"
            out_path = os.path.join(save_dir, out_filename)
            
            tiff.imwrite(out_path, save_img)
            saved_files.append(out_path)
            count += 1

        print(f"[{base_name}] Processed: {count} cells.")
        return saved_files

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def generate_folder_grid(folder_path):
    """
    指定されたフォルダ内の画像からランダムに10枚選んでグリッド画像 (alignment_check.png) を生成する。
    """
    try:
        images = sorted(glob(os.path.join(folder_path, '*.tif')))
        if not images: return

        # 最大10枚ランダム選択
        import random
        selected = random.sample(images, min(len(images), 10))
        
        cols = 5
        rows = math.ceil(len(selected) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if len(selected) == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < len(selected):
                img = tiff.imread(selected[i])
                # RGB表示 (R=Chl, G=Pyr)
                rgb = np.zeros((64, 64, 3))
                if img.ndim == 3 and img.shape[-1] >= 2:
                    rgb[..., 0] = img[..., 0] # Red
                    rgb[..., 1] = img[..., 1] # Green
                ax.imshow(np.clip(rgb, 0, 1))
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(folder_path, "alignment_check.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        # print(f"  Generated check grid for {os.path.basename(folder_path)}")
        
    except Exception as e:
        print(f"Error generating grid for {folder_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create Aligned Dataset V4 (Strict Filter & Centering)")
    parser.add_argument('--input_dir', type=str, required=True, help="Root directory of raw images")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--workers', type=int, default=-1, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # 1. ファイル探索 (再帰的)
    input_files = sorted(glob(os.path.join(args.input_dir, '**', '*.tif'), recursive=True) + 
                         glob(os.path.join(args.input_dir, '**', '*.tiff'), recursive=True))
    
    if not input_files:
        print("No input files found.")
        return

    print(f"Found {len(input_files)} images. Starting processing...")

    # 2. 並列処理で細胞抽出
    # 結果は保存されたファイルパスのリストのリスト
    _ = Parallel(n_jobs=args.workers)(
        delayed(process_single_file)(f, args.input_dir, args.output_dir) 
        for f in input_files
    )
    
    print("\nProcessing complete. Generating confirmation grids...")

    # 3. フォルダごとの確認用画像生成 (v3準拠)
    # 出力ディレクトリを走査して、tifファイルがあるフォルダ全てに対してグリッドを作成
    for root, dirs, files in os.walk(args.output_dir):
        has_tif = any(f.endswith('.tif') for f in files)
        if has_tif:
            generate_folder_grid(root)

    print("All done.")

if __name__ == "__main__":
    main()