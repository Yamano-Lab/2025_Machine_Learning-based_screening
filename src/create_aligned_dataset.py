import os
import numpy as np
import tifffile as tiff
from glob import glob
import argparse
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.transform import resize, rotate
from skimage import exposure
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import math
from joblib import Parallel, delayed

# ==========================================
# 設定
# ==========================================
def align_and_crop_single_image(file_path, output_dir, model_name='2D_versatile_fluo', crop_size=64):
    """
    修正版: 正しい回転ロジックでピレノイドを「下」に揃える
    """
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = StarDist2D.from_pretrained(model_name)

        image = tiff.imread(file_path)
        
        if image.ndim == 3 and image.shape[-1] >= 2:
            red_channel = image[..., 0]
            green_channel = image[..., 1]
            seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
        else:
            return 0 

        normalized_seg = normalize(seg_channel)
        labels, _ = model.predict_instances(normalized_seg)
        props = regionprops(labels)

        p99_red = np.percentile(red_channel, 99.9)
        p99_green = np.percentile(green_channel, 99.9)
        
        count = 0
        
        for i, prop in enumerate(props):
            if prop.area < 200 or prop.area > 8000: continue
            if prop.eccentricity > 0.95: continue
            
            # --- Shape Filtering (New) ---
            # 1. Solidity check (Reject concave/irregular shapes)
            if prop.solidity < 0.9: continue
            
            # 2. Circularity check (Reject complex/elongated shapes)
            # Formula: (4 * pi * Area) / (Perimeter^2)
            circularity = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            if circularity < 0.8: continue

            minr, minc, maxr, maxc = prop.bbox
            h, w = maxr - minr, maxc - minc
            margin = int(max(h, w) * 0.5)
            
            r_start = max(0, minr - margin)
            r_end = min(red_channel.shape[0], maxr + margin)
            c_start = max(0, minc - margin)
            c_end = min(red_channel.shape[1], maxc + margin)
            
            crop_red = red_channel[r_start:r_end, c_start:c_end]
            crop_green = green_channel[r_start:r_end, c_start:c_end]
            crop_mask = labels[r_start:r_end, c_start:c_end] == prop.label

            crop_red = crop_red * crop_mask
            crop_green = crop_green * crop_mask

            # --- アライメント修正箇所 ---
            m = regionprops(crop_mask.astype(int), intensity_image=crop_red)[0]
            cy, cx = m.centroid
            
            if np.max(crop_green) < p99_green * 0.1: continue

            mg = regionprops(crop_mask.astype(int), intensity_image=crop_green)[0]
            py, px = mg.weighted_centroid
            
            # ベクトル: 細胞中心 -> ピレノイド
            # 画像座標系: yは下向きに増加、xは右向きに増加
            dy, dx = py - cy, px - cx
            
            if dy**2 + dx**2 < 2.0:
                angle = 0
            else:
                # atan2(y, x): (0,1)=0度, (1,0)=90度, (0,-1)=180度, (-1,0)=-90度
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                
                # 目標: ベクトルを (1, 0) つまり 90度（真下）に向けたい
                # 回転量 = 現在の角度 - 目標角度 (反時計回りが正の操作の場合...逆か)
                # 検証:
                #  右(0度) -> 下(90度)に行きたい -> 時計回りに90度 -> rotate(-90)
                #  式: 0 - 90 = -90 (OK)
                #  左(180度) -> 下(90度)に行きたい -> 反時計に90度 -> rotate(90)
                #  式: 180 - 90 = 90 (OK)
                #  上(-90度) -> 下(90度)に行きたい -> 180度 -> rotate(180)
                #  式: -90 - 90 = -180 (OK)
                #
                # 結論: angle_deg - 90 が正解
                angle = angle_deg - 90

            # 回転
            rotated_red = rotate(crop_red, angle, resize=False, preserve_range=True)
            rotated_green = rotate(crop_green, angle, resize=False, preserve_range=True)
            
            # クロップ & リサイズ
            center_y, center_x = rotated_red.shape[0] // 2, rotated_red.shape[1] // 2
            size = max(h, w)
            y1 = max(0, center_y - size // 2)
            y2 = min(rotated_red.shape[0], center_y + size // 2)
            x1 = max(0, center_x - size // 2)
            x2 = min(rotated_red.shape[1], center_x + size // 2)
            
            final_red = rotated_red[y1:y2, x1:x2]
            final_green = rotated_green[y1:y2, x1:x2]
            
            final_red = resize(final_red, (crop_size, crop_size), anti_aliasing=True)
            final_green = resize(final_green, (crop_size, crop_size), anti_aliasing=True)
            
            final_red = np.clip(final_red / p99_red, 0, 1)
            final_green = np.clip(final_green / p99_green, 0, 1)

            save_img = np.stack([final_red, final_green], axis=-1).astype(np.float32)
            save_name = f"{base_name}_cell_{i:04d}.tif"
            tiff.imwrite(os.path.join(output_dir, save_name), save_img)
            
            count += 1
            
        return count
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/raw_images')
    parser.add_argument('--output_dir', type=str, default='./data/aligned_dataset_v2') # 出力先を変更
    parser.add_argument('--workers', type=int, default=-1)
    args = parser.parse_args()
    
    # 既存データのクリーンアップが必要ならここで行うか、別フォルダに出力する
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_paths = sorted(glob(os.path.join(args.input_dir, '*.tif')) + glob(os.path.join(args.input_dir, '*.tiff')))
    print(f"Starting corrected alignment processing for {len(file_paths)} images...")
    
    counts = Parallel(n_jobs=args.workers, verbose=10)(
        delayed(align_and_crop_single_image)(fp, args.output_dir) for fp in file_paths
    )
    
    print(f"\nCorrection Complete. Extracted {sum(counts)} aligned cells.")
    print(f"Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()