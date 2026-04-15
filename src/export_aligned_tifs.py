import os
import argparse
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import shutil

def generate_annotated_images(input_dir, output_dir, num_images_per_folder):
    """
    指定されたフォルダから画像を再帰的に検索し、注釈付きの
    マゼンタ＆グリーン画像（PNG）を生成して保存する。

    Args:
        input_dir (str): 画像が格納されている親ディレクトリ。
        output_dir (str): 画像を保存するディレクトリ。
        num_images_per_folder (int): 各サブフォルダから処理する画像の数。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    all_image_files = list(input_path.rglob('*.tif')) + \
                      list(input_path.rglob('*.png')) + \
                      list(input_path.rglob('*.jpg')) + \
                      list(input_path.rglob('*.jpeg'))
                      
    images_by_dir = {}
    for f in all_image_files:
        parent_dir = f.parent
        if parent_dir not in images_by_dir:
            images_by_dir[parent_dir] = []
        images_by_dir[parent_dir].append(f)

    if not images_by_dir:
        print(f"No images found in subdirectories of '{input_dir}'")
        return

    print(f"Found {len(images_by_dir)} directories with images. Processing...")

    for strain_dir in sorted(images_by_dir.keys()):
        image_files = images_by_dir[strain_dir]
        
        try:
            relative_path = strain_dir.relative_to(input_path)
        except ValueError:
            relative_path = Path(strain_dir.name)

        output_strain_dir = output_path / relative_path
        output_strain_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Processing directory: {relative_path}")

        files_to_process = sorted(image_files)[:num_images_per_folder]
        print(f"    Generating {len(files_to_process)} annotated images...")

        for image_file in files_to_process:
            output_filename = output_strain_dir / f"{image_file.stem}.png"

            try:
                if "alignment_check" in image_file.name or "filter_diagnosis" in image_file.name:
                    print(f"    Skipping diagnostic file: {image_file.name}")
                    continue

                img = tifffile.imread(str(image_file))
                
                display_img = None
                if img.ndim == 3 and img.shape[2] == 2:
                    h, w, c = img.shape
                    rgb = np.zeros((h, w, 3), dtype=np.float32)
                    
                    red_channel_data = img[..., 0]
                    green_channel_data = img[..., 1]

                    # Red -> Magenta (Red + Blue)
                    rgb[..., 0] = red_channel_data  # Red component
                    rgb[..., 2] = red_channel_data  # Blue component
                    
                    # Green channel remains
                    rgb[..., 1] = green_channel_data

                    min_val, max_val = rgb.min(), rgb.max()
                    if max_val > min_val:
                        display_img = (rgb - min_val) / (max_val - min_val)
                    else:
                        display_img = rgb
                
                elif img.ndim == 3 and img.shape[2] in [3, 4]:
                    display_img = img
                elif img.ndim == 2:
                    display_img = np.stack([img]*3, axis=-1)
                else:
                    print(f"    WARNING: Skipping {image_file.name}, unsupported shape {img.shape}")
                    continue
                
                if np.isnan(display_img).any() or np.isinf(display_img).any():
                    print(f"    WARNING: NaN/Inf detected in {image_file.name}, skipping.")
                    continue
                
                if np.issubdtype(display_img.dtype, np.floating) and (display_img.min() < 0.0 or display_img.max() > 1.0):
                    min_val, max_val = display_img.min(), display_img.max()
                    if max_val > min_val:
                         display_img = (display_img - min_val) / (max_val - min_val)
                elif np.issubdtype(display_img.dtype, np.integer) and (display_img.min() < 0 or display_img.max() > 255):
                     min_val, max_val = display_img.min(), display_img.max()
                     if max_val > min_val:
                        display_img = ((display_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                plt.figure(figsize=(4, 4))
                plt.imshow(display_img)
                plt.title(f"{image_file.name}\nMin:{img.min():.2f} Max:{img.max():.2f}", fontsize=8)
                plt.axis('off')
                plt.savefig(str(output_filename), bbox_inches='tight', dpi=150)
                plt.close()

            except Exception as e:
                print(f"    Error processing {image_file}: {e}")

    print(f"\nImage generation complete. Files saved in '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate annotated inspection images (Magenta/Green) from subfolders.')
    parser.add_argument('input_dir', type=str, help='Input directory containing strain subfolders.')
    parser.add_argument('output_dir', type=str, help='Output directory to save images.')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to generate from each folder.')

    args = parser.parse_args()

    generate_annotated_images(args.input_dir, args.output_dir, args.num_images)
