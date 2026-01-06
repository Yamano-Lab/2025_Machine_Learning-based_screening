import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from glob import glob
from tensorflow.keras.models import load_model
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
from stardist.models import StarDist2D
import argparse

# ==========================================
# 設定
# ==========================================
# デフォルト値（引数で上書き可能）
DEFAULT_INPUT_DIR = "data/raw_images"
DEFAULT_MODEL_DIR = "results/Models/20260105_1654" 

def parse_args():
    parser = argparse.ArgumentParser(description="Generate reconstruction images from trained CAE model.")
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, help='Directory containing raw TIF images.')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, help='Directory containing the trained model.')
    return parser.parse_args()

def extract_cells(file_paths, stardist_model, max_cells=20):
    sample_cells = []
    
    for file_path in file_paths:
        try:
            image = tiff.imread(file_path)
            
            # --- Dual Channel Extraction Logic ---
            # Attempt to extract Red (0) and Green (1) channels
            if image.ndim == 3 and image.shape[-1] >= 2:
                red_channel = image[..., 0]
                green_channel = image[..., 1]
                # Segmentation channel (usually 2 or 1)
                seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
            else:
                # Fallback for single channel (unlikely for this script but needed for safety)
                # If only 1 channel, duplicate it for compatibility or handle separately
                red_channel = image
                green_channel = image
                seg_channel = image

            normalized_seg = normalize(seg_channel)
            labels, _ = stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            max_red = np.max(red_channel) if np.max(red_channel) > 0 else 1.0
            max_green = np.max(green_channel) if np.max(green_channel) > 0 else 1.0

            for prop in props:
                if prop.area < 200 or prop.area > 8000: continue
                minr, minc, maxr, maxc = prop.bbox
                
                # Crop
                cell_red = red_channel[minr:maxr, minc:maxc]
                cell_green = green_channel[minr:maxr, minc:maxc]
                
                # Quality check (Green)
                if np.mean(cell_green) < 0.5: continue
                
                # Resize & Normalize
                # Red
                cell_red_float = cell_red / max_red
                cell_red_eq = exposure.equalize_adapthist(cell_red_float, clip_limit=0.02)
                cell_red_resized = resize(cell_red_eq, (64, 64), anti_aliasing=True)
                
                # Green
                cell_green_float = cell_green / max_green
                cell_green_eq = exposure.equalize_adapthist(cell_green_float, clip_limit=0.02)
                cell_green_resized = resize(cell_green_eq, (64, 64), anti_aliasing=True)
                
                # Stack: (64, 64, 2)
                cell_combined = np.stack([cell_red_resized, cell_green_resized], axis=-1)
                sample_cells.append(cell_combined)
                
                if len(sample_cells) >= max_cells: return sample_cells
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue
            
    return sample_cells

def visualize_2ch(image_data):
    """
    Visualize (64, 64, 2) as an RGB image.
    R=Red Ch, G=Green Ch, B=0
    """
    h, w, c = image_data.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if c >= 1: rgb[..., 0] = image_data[..., 0] # Red
    if c >= 2: rgb[..., 1] = image_data[..., 1] # Green
    # Blue remains 0
    return np.clip(rgb, 0, 1)

def main():
    args = parse_args()
    
    # Handle "latest" keyword or explicit path
    model_dir = args.model_dir
    input_dir = args.input_dir

    print(f"Loading model from: {model_dir}")
    
    # 1. 保存されたモデルの読み込み
    model_path = os.path.join(model_dir, 'best_autoencoder.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        # Load model. Note: Custom loss might be needed if compiling, but compile=False is safer for inference
        autoencoder = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check model input shape to decide on processing
    input_shape = autoencoder.input_shape
    print(f"Model Input Shape: {input_shape}")
    expected_channels = input_shape[-1]

    # 2. 画像データの再読み込み
    print("Loading a few images for visualization...")
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    file_paths = sorted(glob(os.path.join(input_dir, '*.tif')) + glob(os.path.join(input_dir, '*.tiff')))
    
    if not file_paths:
        print("Error: No images found.")
        return

    # Extract cells (Dual Channel)
    sample_cells = extract_cells(file_paths, stardist_model, max_cells=20)

    if len(sample_cells) == 0:
        print("Error: No cells extracted.")
        return

    print(f"Extracted {len(sample_cells)} cells.")
    
    # Prepare Input
    X = np.array(sample_cells).astype('float32') # (N, 64, 64, 2)
    
    # If model expects 1 channel (old model) but we have 2, take only Green (index 1)
    if expected_channels == 1 and X.shape[-1] == 2:
        print("Warning: Model expects 1 channel but data has 2. Using Green channel only.")
        X = X[..., 1:2]
    # If model expects 2 channels and data has 1 (unlikely here but possible), duplicate?
    elif expected_channels == 2 and X.shape[-1] == 1:
        print("Warning: Model expects 2 channels but data has 1. Duplicating.")
        X = np.concatenate([X, X], axis=-1)

    # 推論
    reconstructed = autoencoder.predict(X, verbose=0)
    
    # プロット作成 (Dual Channel Visualization)
    n_samples = min(10, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    if n_samples == 1: axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        # Determine how to visualize based on channels
        if X.shape[-1] == 2:
            img_in = visualize_2ch(X[idx])
            img_out = visualize_2ch(reconstructed[idx])
        else:
            img_in = X[idx].squeeze()
            img_out = reconstructed[idx].squeeze()
            
        # Original
        axes[0, i].imshow(img_in, cmap='gray' if X.shape[-1]==1 else None)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(img_out, cmap='gray' if X.shape[-1]==1 else None)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(model_dir, 'reconstruction_samples_dual_check.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nImage saved to: {save_path}")
    # plt.show() # CLI実行時はshowしないことが多い

if __name__ == "__main__":
    main()
