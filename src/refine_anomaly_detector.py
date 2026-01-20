import argparse
import os
import pickle
import numpy as np
import tifffile as tiff
from glob import glob
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def load_wt_images(data_dir):
    """
    Finds the 'WT' directory within data_dir and loads all .tif/.tiff images recursively.
    """
    # 1. Find WT folder
    wt_dir = None
    # Check if data_dir itself is WT
    if os.path.basename(os.path.normpath(data_dir)).upper() == 'WT':
        wt_dir = data_dir
    else:
        # Search immediate subdirectories
        for d in os.listdir(data_dir):
            if d.upper() == 'WT' and os.path.isdir(os.path.join(data_dir, d)):
                wt_dir = os.path.join(data_dir, d)
                break
    
    if wt_dir is None:
        print(f"Warning: Could not find a 'WT' folder in {data_dir}. Searching entire directory for images...")
        wt_dir = data_dir

    print(f"Loading WT images from: {wt_dir}")
    
    # 2. Collect files recursively
    patterns = [
        os.path.join(wt_dir, '**', '*.tif'),
        os.path.join(wt_dir, '**', '*.tiff')
    ]
    file_paths = []
    for p in patterns:
        file_paths.extend(glob(p, recursive=True))
    
    file_paths = sorted(list(set(file_paths)))
    print(f"Found {len(file_paths)} images.")
    
    images = []
    for fp in tqdm(file_paths, desc="Reading images"):
        try:
            img = tiff.imread(fp)
            # Expecting (64, 64, 2)
            if img.ndim == 3 and img.shape == (64, 64, 2):
                images.append(img)
            else:
                # pass silently or warn?
                pass
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            
    return np.array(images).astype('float32')

def main():
    parser = argparse.ArgumentParser(description="Refine Anomaly Detector (SVM & IF) using pre-trained Encoder.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the aligned dataset directory containing a WT folder.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the directory containing the trained 'encoder.keras'. Output models will be saved here.")
    args = parser.parse_args()

    # 1. Load Data
    X_train = load_wt_images(args.data_dir)
    if len(X_train) == 0:
        print("No valid WT images found. Exiting.")
        return
    
    print(f"Training data shape: {X_train.shape}")

    # 2. Feature Extraction
    encoder_path = os.path.join(args.model_dir, 'encoder.keras')
    if not os.path.exists(encoder_path):
        print(f"Encoder not found at {encoder_path}")
        return
        
    print("Loading encoder...")
    encoder = load_model(encoder_path, compile=False)
    
    print("Extracting features...")
    # Process in batches to avoid OOM
    batch_size = 32
    features_list = []
    for i in tqdm(range(0, len(X_train), batch_size), desc="Encoding"):
        batch = X_train[i:i+batch_size]
        feats = encoder.predict(batch, verbose=0)
        # Flatten: (B, 8, 8, 32) -> (B, 2048)
        feats_flat = feats.reshape(len(feats), -1)
        features_list.append(feats_flat)
        
    features = np.concatenate(features_list, axis=0)
    print(f"Raw feature shape: {features.shape}")

    # 3. Preprocessing Pipeline
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print("Fitting PCA (n_components=0.95)...")
    pca = PCA(n_components=0.95, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print(f"PCA reduced dimensions: {features.shape[1]} -> {features_pca.shape[1]}")

    # 4. Train Models
    # SVM
    print("Training Optimized One-Class SVM...")
    # nu=0.01: Allow 1% of WT to be outliers (loose boundary)
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    svm.fit(features_pca)
    
    # Isolation Forest
    print("Training Isolation Forest (Comparison)...")
    iso_forest = IsolationForest(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1)
    iso_forest.fit(features_pca)

    # 5. Save Artifacts
    print("Saving models to", args.model_dir)
    
    save_map = {
        'scaler_v2.pkl': scaler,
        'pca_v2.pkl': pca,
        'detector_svm_v2.pkl': svm,
        'detector_if_v2.pkl': iso_forest
    }
    
    for filename, obj in save_map.items():
        path = os.path.join(args.model_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved {filename}")

    print("\nRefinement Complete. You can now use these V2 models in the screening pipeline.")

if __name__ == "__main__":
    main()
