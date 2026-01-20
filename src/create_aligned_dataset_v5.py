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
from skimage.segmentation import mark_boundaries
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from joblib import Parallel, delayed

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')

def process_single_file(file_path, input_root, output_root, solidity_th, circularity_th, eccentricity_th, is_first_file=False):
    """
    厳格な基準で細胞を抽出・整列して保存する (v5.2: 属性エラー修正版)。
    """
    try:
        # パス設定
        rel_path = os.path.relpath(file_path, input_root)
        save_dir = os.path.join(output_root, os.path.dirname(rel_path))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(save_dir, exist_ok=True)
        
        # GPU無効化
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = StarDist2D.from_pretrained('2D_versatile_fluo')

        image = tiff.imread(file_path)
        
        # --- 画像形状の自動修正 ---
        if image.ndim == 3 and image.shape[0] <= 5 and image.shape[2] > 50:
            if is_first_file:
                print(f"  [Info] Transposing image from {image.shape} (C,H,W) to (H,W,C)")
            image = np.transpose(image, (1, 2, 0))
        
        if is_first_file:
            print(f"  [Debug] File: {os.path.basename(file_path)}")
            print(f"  [Debug] Shape loaded: {image.shape}")

        # チャンネル分解
        if image.ndim == 3 and image.shape[-1] >= 2:
            red_channel = image[..., 0]
            green_channel = image[..., 1]
            seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
        elif image.ndim == 2:
            return [], {}
        else:
            return [], {}

        normalized_seg = normalize(seg_channel)
        labels, _ = model.predict_instances(normalized_seg)
        props = regionprops(labels)
        
        if is_first_file:
            print(f"  [Debug] Detected cells (raw): {len(props)}")

        p99_red = np.percentile(red_channel, 99.9)
        p99_green = np.percentile(green_channel, 99.9)
        
        saved_files = []
        H, W = red_channel.shape
        count = 0
        
        stats = {'total': len(props), 'saved': 0, 'border': 0, 'size': 0, 'eccentricity': 0, 'solidity': 0, 'circularity': 0}
        diagnosis_map = {}

        for prop in props:
            # 1. 見切れ除去
            minr, minc, maxr, maxc = prop.bbox
            if minr <= 5 or minc <= 5 or maxr >= H - 5 or maxc >= W - 5:
                stats['border'] += 1
                diagnosis_map[prop.label] = 'Border'
                continue

            # 2. 形状フィルタ
            if prop.area < 200 or prop.area > 8000:
                stats['size'] += 1
                diagnosis_map[prop.label] = 'Size'
                continue
            
            if prop.eccentricity > eccentricity_th:
                stats['eccentricity'] += 1
                diagnosis_map[prop.label] = 'Eccentricity'
                continue

            if prop.solidity < solidity_th:
                stats['solidity'] += 1
                diagnosis_map[prop.label] = 'Solidity'
                continue
            
            circularity = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            if circularity < circularity_th:
                stats['circularity'] += 1
                diagnosis_map[prop.label] = 'Circularity'
                continue

            diagnosis_map[prop.label] = 'Accepted'

            # --- センタリングと切り出し ---
            cy, cx = map(int, prop.centroid)
            
            # ★修正箇所: bbox_height/bbox_width ではなく計算値を使用
            bbox_h = maxr - minr
            bbox_w = maxc - minc
            
            crop_size_raw = int(max(bbox_h, bbox_w) * 1.5)
            crop_size_raw = max(crop_size_raw, 64)
            half_size = crop_size_raw // 2

            pad_amount = crop_size_raw
            padded_red = np.pad(red_channel, pad_amount, mode='constant')
            padded_green = np.pad(green_channel, pad_amount, mode='constant')
            padded_mask = np.pad(labels == prop.label, pad_amount, mode='constant')
            
            cy_pad, cx_pad = cy + pad_amount, cx + pad_amount
            
            crop_red = padded_red[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]
            crop_green = padded_green[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]
            crop_mask = padded_mask[cy_pad - half_size : cy_pad + half_size, cx_pad - half_size : cx_pad + half_size]

            crop_red = crop_red * crop_mask
            crop_green = crop_green * crop_mask

            # アライメント
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
            angle = 0 if dy**2 + dx**2 < 2.0 else math.degrees(math.atan2(dy, dx)) - 90

            rotated_red = rotate(crop_red, angle, resize=False, preserve_range=True)
            rotated_green = rotate(crop_green, angle, resize=False, preserve_range=True)
            
            final_size = 64
            center_y, center_x = rotated_red.shape[0] // 2, rotated_red.shape[1] // 2
            y1, x1 = center_y - final_size // 2, center_x - final_size // 2
            
            final_red = resize(rotated_red, (final_size, final_size), anti_aliasing=True)
            final_green = resize(rotated_green, (final_size, final_size), anti_aliasing=True)
            
            final_red = np.clip(final_red / p99_red, 0, 1)
            final_green = np.clip(final_green / p99_green, 0, 1)

            save_img = np.stack([final_red, final_green], axis=-1).astype(np.float32)
            out_path = os.path.join(save_dir, f"{base_name}_cell_{count:04d}.tif")
            tiff.imwrite(out_path, save_img)
            saved_files.append(out_path)
            count += 1

        stats['saved'] = count
        
        if is_first_file:
            create_diagnosis_image(red_channel, labels, diagnosis_map, output_root)

        return saved_files, stats

    except Exception as e:
        if is_first_file: print(f"  [Error] Processing failed: {e}")
        return [], {}

def create_diagnosis_image(image, labels, diagnosis_map, output_dir):
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        overlay = np.zeros(image.shape + (3,), dtype=float)
        
        from skimage.segmentation import find_boundaries
        
        for label_id, status in diagnosis_map.items():
            mask = (labels == label_id)
            color = [0, 1, 0] if status == 'Accepted' else [1, 0, 0]
            boundaries = find_boundaries(mask)
            overlay[boundaries] = color
            
            if np.any(mask):
                y, x = regionprops(mask.astype(int))[0].centroid
                plt.text(x, y, status[:4], color='yellow' if status=='Accepted' else 'red', fontsize=6, ha='center')

        plt.imshow(overlay, alpha=0.5)
        plt.title("Filter Diagnosis (Green=Accepted, Red=Rejected)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "filter_diagnosis.png"), dpi=150)
        plt.close()
        print(f"  [Diagnosis] Saved filter diagnosis image to {output_dir}/filter_diagnosis.png")
    except Exception as e:
        print(f"Failed to create diagnosis image: {e}")

def generate_folder_grid(folder_path):
    try:
        images = sorted(glob(os.path.join(folder_path, '*.tif')))
        if not images: return
        import random
        selected = random.sample(images, min(len(images), 10))
        cols = 5; rows = math.ceil(len(selected) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if len(selected) == 1: axes = np.array([axes])
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < len(selected):
                img = tiff.imread(selected[i])
                rgb = np.zeros((64, 64, 3))
                if img.ndim==3: rgb[..., 0]=img[..., 0]; rgb[..., 1]=img[..., 1]
                ax.imshow(np.clip(rgb, 0, 1)); ax.axis('off')
            else: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "alignment_check.png"), dpi=150)
        plt.close()
    except: pass

def main():
    parser = argparse.ArgumentParser(description="Create Aligned Dataset V5.2 (Fix Attribute Error)")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=-1)
    
    parser.add_argument('--solidity', type=float, default=0.9)
    parser.add_argument('--circularity', type=float, default=0.8)
    parser.add_argument('--eccentricity', type=float, default=0.95)
    
    args = parser.parse_args()
    
    input_files = sorted(glob(os.path.join(args.input_dir, '**', '*.tif'), recursive=True) + 
                         glob(os.path.join(args.input_dir, '**', '*.tiff'), recursive=True))
    
    if not input_files:
        print(f"[ERROR] No TIF files found in '{args.input_dir}'")
        return

    print(f"Found {len(input_files)} images.")
    print(f"Filters: Solidity > {args.solidity}, Circularity > {args.circularity}, Eccentricity < {args.eccentricity}")
    print("Processing...")

    results = Parallel(n_jobs=args.workers)(
        delayed(process_single_file)(f, args.input_dir, args.output_dir, args.solidity, args.circularity, args.eccentricity, i==0) 
        for i, f in enumerate(input_files)
    )
    
    total_cells = sum(r[1].get('total', 0) for r in results)
    total_saved = sum(r[1].get('saved', 0) for r in results)
    
    print("\n" + "="*40)
    print(f"Total Detected: {total_cells}")
    print(f"Total Saved:    {total_saved}")
    print("-" * 20)
    
    if total_saved == 0:
        print("[WARNING] No cells saved.")
        print(f"  Border:      {sum(r[1].get('border', 0) for r in results)}")
        print(f"  Size:        {sum(r[1].get('size', 0) for r in results)}")
        print(f"  Solidity:    {sum(r[1].get('solidity', 0) for r in results)}")
        print(f"  Circularity: {sum(r[1].get('circularity', 0) for r in results)}")
        print(f"  Eccentricity:{sum(r[1].get('eccentricity', 0) for r in results)}")
    else:
        for root, dirs, files in os.walk(args.output_dir):
            if any(f.endswith('.tif') for f in files): generate_folder_grid(root)
        print("Grids generated.")

if __name__ == "__main__":
    main()