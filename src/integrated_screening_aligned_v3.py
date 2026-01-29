# coding: utf-8
import argparse
import os
import pickle
from datetime import datetime
from glob import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
import umap.umap_ as umap
from csbdeep.utils import normalize
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from skimage.measure import regionprops
from skimage.transform import resize, rotate
from stardist.models import StarDist2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, MaxPooling2D

# --- New imports for extended analysis ---
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
try:
    import phate
except ImportError:
    phate = None
try:
    import scanpy as sc
    import anndata
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


class MutantScreeningPipeline:
    """
    変異株スクリーニングの統合パイプライン (Aligned Model 対応版) v3
    - 査読対応版: 
        1. 面積フィルタリングを緩和し、サイズ異常をスコア化 (Size Z-score)
        2. クラスタリング/UMAP入力を再構成誤差(Residuals)から潜在特徴量(Latent Features)に変更
        3. アライメントのロバスト性向上 (ピレノイド不鮮明時は細胞長軸で補正)
        4. クラスタリング手法をK-Means(k=10)からLeiden(Graph-based)へ変更
    """

    def __init__(self, model_dir, use_prealigned=False):
        self.model_dir = model_dir
        self.use_prealigned = use_prealigned
        
        # --- Visual Style Setup (Okabe-Ito & Publication Ready) ---
        self.okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        sns.set_palette(self.okabe_ito)
        sns.set_style("ticks")
        plt.rcParams.update({
            'font.size': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.autolayout': False
        })
        
        self.load_trained_models()

    def load_trained_models(self):
        """訓練済みAIモデルと関連ファイルを読み込む"""
        print("Loading trained models...")
        try:
            self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'), compile=False)
            
            # --- Dynamically extract Decoder ---
            maxpool_idx = -1
            for i, layer in enumerate(self.autoencoder.layers):
                if isinstance(layer, MaxPooling2D):
                    maxpool_idx = i
            
            if maxpool_idx != -1:
                decoder_layers = self.autoencoder.layers[maxpool_idx+1:]
                decoder_input = Input(shape=(8, 8, 32))
                x = decoder_input
                for layer in decoder_layers:
                    x = layer(x)
                self.decoder = Model(decoder_input, x)
                print("  Decoder extracted and reconstructed successfully.")
            else:
                pass

            self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'), compile=False)
            
            # --- Load Scaler ---
            scaler_path_v2 = os.path.join(self.model_dir, 'scaler_v2.pkl')
            scaler_path_v1 = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path_v2):
                print(f"  Loading Scaler V2 from {scaler_path_v2}")
                with open(scaler_path_v2, 'rb') as f: self.scaler = pickle.load(f)
            else:
                print(f"  Loading Scaler V1 from {scaler_path_v1}")
                with open(scaler_path_v1, 'rb') as f: self.scaler = pickle.load(f)

            # --- Load PCA ---
            pca_path_v2 = os.path.join(self.model_dir, 'pca_v2.pkl')
            pca_path_v1 = os.path.join(self.model_dir, 'pca.pkl')
            if os.path.exists(pca_path_v2):
                print(f"  Loading PCA V2 from {pca_path_v2}")
                with open(pca_path_v2, 'rb') as f: self.pca = pickle.load(f)
            else:
                print(f"  Loading PCA V1 from {pca_path_v1}")
                with open(pca_path_v1, 'rb') as f: self.pca = pickle.load(f)

            # --- Load Detector ---
            det_path_v2 = os.path.join(self.model_dir, 'detector_svm_v2.pkl')
            det_path_v1 = os.path.join(self.model_dir, 'detector_conservative.pkl')
            
            if os.path.exists(det_path_v2):
                print(f"  Loading Detector V2 (SVM) from {det_path_v2}")
                with open(det_path_v2, 'rb') as f: self.detector_conservative = pickle.load(f)
            elif os.path.exists(det_path_v1):
                print(f"  Loading Detector V1 from {det_path_v1}")
                with open(det_path_v1, 'rb') as f: self.detector_conservative = pickle.load(f)
            else:
                print("  [Warning] No anomaly detector found (using MSE only).")
                self.detector_conservative = None
                
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models from {self.model_dir}: {e}")
            raise

    def extract_quality_cells(self, image_path, enhance_contrast=False):
        """
        単一のTIF画像から品質基準を満たし、アライメント済みの細胞画像を抽出する。
        v3変更点: 
        - アライメントのフォールバック (ピレノイドがない場合は細胞長軸で整列)
        Returns: [(raw, aligned, area, align_method), ...] 
        """
        try:
            if self.use_prealigned:
                try:
                    image = tiff.imread(image_path)
                    if image.shape == (64, 64, 2):
                        # Calculate area for pre-aligned (Red channel > 0.1)
                        area = np.sum(image[..., 0] > 0.1)
                        return [(image, image, area, 'prealigned')]
                    else:
                        return []
                except Exception as e:
                    print(f"Error reading pre-aligned image {image_path}: {e}")
                    return []

            image = tiff.imread(image_path)
            if image.ndim == 3 and image.shape[-1] >= 2:
                red_channel = image[..., 0]
                green_channel = image[..., 1]
                seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
            else:
                return []

            normalized_seg = normalize(seg_channel)
            labels, _ = self.stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            quality_cells = []
            
            p99_red = np.percentile(red_channel, 99.9)
            p99_green = np.percentile(green_channel, 99.9)
            
            H, W = red_channel.shape
            
            for prop in props:
                # 1. Strict Border Rejection (5px margin)
                minr, minc, maxr, maxc = prop.bbox
                if minr < 5 or minc < 5 or maxr > H - 5 or maxc > W - 5: continue

                # 2. Relaxed Shape Filtering (Quality Control) - Keep very small/large for size analysis
                # But filter extreme noise (e.g. area < 50)
                if prop.area < 50: continue 
                
                # Still filter very non-cell like shapes
                if prop.eccentricity > 0.98: continue
                # if prop.solidity < 0.8: continue # Relaxed

                # 3. Centroid-based Padding & Crop
                cy, cx = map(int, prop.centroid)
                
                bbox_h = maxr - minr
                bbox_w = maxc - minc
                size = int(max(bbox_h, bbox_w) * 1.5)
                size = max(size, 64)
                half_size = size // 2
                
                r_start = cy - half_size
                c_start = cx - half_size
                r_end = r_start + size
                c_end = c_start + size
                
                crop_red = np.zeros((size, size), dtype=red_channel.dtype)
                crop_green = np.zeros((size, size), dtype=green_channel.dtype)
                crop_mask = np.zeros((size, size), dtype=bool)
                
                r_start_clamped = max(0, r_start)
                r_end_clamped = min(H, r_end)
                c_start_clamped = max(0, c_start)
                c_end_clamped = min(W, c_end)
                
                dr_start = r_start_clamped - r_start
                dr_end = dr_start + (r_end_clamped - r_start_clamped)
                dc_start = c_start_clamped - c_start
                dc_end = dc_start + (c_end_clamped - c_start_clamped)
                
                if dr_end > dr_start and dc_end > dc_start:
                    crop_red[dr_start:dr_end, dc_start:dc_end] = red_channel[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped]
                    crop_green[dr_start:dr_end, dc_start:dc_end] = green_channel[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped]
                    mask_slice = (labels[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped] == prop.label)
                    crop_mask[dr_start:dr_end, dc_start:dc_end] = mask_slice
                else:
                    continue

                crop_red = crop_red * crop_mask
                crop_green = crop_green * crop_mask

                # 4. Alignment Logic (Robust)
                align_method = 'pyrenoid'
                angle = 0
                
                try:
                    m = regionprops(crop_mask.astype(int), intensity_image=crop_red)[0]
                    cy_crop, cx_crop = m.centroid
                    
                    # Check Pyrenoid Signal
                    max_green = np.max(crop_green)
                    if max_green >= p99_green * 0.1:
                        # Strategy A: Pyrenoid-based (Standard)
                        mg = regionprops(crop_mask.astype(int), intensity_image=crop_green)[0]
                        py, px = mg.weighted_centroid
                        dy, dx = py - cy_crop, px - cx_crop
                        if dy**2 + dx**2 >= 2.0:
                            angle_rad = math.atan2(dy, dx)
                            angle_deg = math.degrees(angle_rad)
                            angle = angle_deg - 90
                    else:
                        # Strategy B: Cell Axis-based (Fallback for Pyrenoid-less mutants)
                        # Align major axis to vertical
                        align_method = 'axis'
                        orientation = m.orientation # -pi/2 to pi/2
                        angle = -math.degrees(orientation)
                        
                except IndexError:
                    continue

                rotated_red = rotate(crop_red, angle, resize=False, preserve_range=True)
                rotated_green = rotate(crop_green, angle, resize=False, preserve_range=True)
                
                # Final Resize to 64x64
                final_crop_size = 64
                final_red = resize(rotated_red, (final_crop_size, final_crop_size), anti_aliasing=True)
                final_green = resize(rotated_green, (final_crop_size, final_crop_size), anti_aliasing=True)
                
                # 5. Normalize (99.9 percentile)
                final_red = np.clip(final_red / p99_red, 0, 1)
                final_green = np.clip(final_green / p99_green, 0, 1)

                aligned_cell = np.stack([final_red, final_green], axis=-1).astype(np.float32)
                
                # Area Calculation
                area = np.sum(final_red > 0.1)
                
                quality_cells.append((aligned_cell, aligned_cell, area, align_method))
                
            return quality_cells
        except Exception as e:
            print(f"Error extracting cells from {os.path.basename(image_path)}: {e}")
            return []

    def analyze_size_distribution(self, all_data, output_dir):
        """
        WTの面積分布に基づいてサイズ異常をスコア化する（極端なゴミ以外は除外しない）。
        
        Args:
            all_data: list of dicts
            output_dir: Output directory path
            
        Returns:
            all_data: with 'size_z_score' added
            kept_data: filtered list (removing only extreme garbage)
        """
        print("\n--- Analyzing Size Distribution (Z-score Calculation) ---")
        
        # 1. Extract WT areas
        wt_areas = [d['area'] for d in all_data if 'WT' in d['sample'].upper() or d['sample'].upper() == 'WT']
        
        if not wt_areas:
            print("  [Warning] WT samples not found. Skipping size scoring.")
            for d in all_data: d['size_z_score'] = 0.0
            return all_data, all_data
            
        wt_areas = np.array(wt_areas)
        wt_mean = np.mean(wt_areas)
        wt_std = np.std(wt_areas)
        
        print(f"  WT Area Stats: Mean={wt_mean:.1f}, Std={wt_std:.1f}")
        
        # 2. Hard Filtering (Garbage Removal only)
        # Assuming 64x64 images, max area ~4096. 
        # Area < 50 is likely noise. Area > 3500 is likely full frame/clump.
        min_area_thresh = 50
        max_area_thresh = 3800
        
        kept_data = []
        removed_summary = {}
        
        for d in all_data:
            area = d['area']
            # Calculate Z-score
            d['size_z_score'] = (area - wt_mean) / (wt_std + 1e-6)
            
            # Hard filter for extreme noise
            if min_area_thresh <= area <= max_area_thresh:
                kept_data.append(d)
            else:
                sample = d['sample']
                removed_summary[sample] = removed_summary.get(sample, 0) + 1

        print(f"  Total Cells Before: {len(all_data)}")
        print(f"  Total Cells Kept:   {len(kept_data)}")
        print(f"  Garbage Removed:    {len(all_data) - len(kept_data)} (Extremely small/large)")
        
        # 3. Visualization
        plot_data = pd.DataFrame([{'sample': d['sample'], 'area': d['area'], 'type': 'WT' if 'WT' in d['sample'].upper() else 'Mutant'} for d in kept_data])
        
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=plot_data[plot_data['type']=='WT'], x='area', fill=True, color='gray', alpha=0.3, label='WT Distribution')
        # Plot a few mutants if too many
        mutants = plot_data[plot_data['type']=='Mutant']['sample'].unique()[:5]
        for m in mutants:
            sns.kdeplot(data=plot_data[plot_data['sample']==m], x='area', label=m, linewidth=1)
            
        plt.title('Cell Area Distribution (WT vs Mutants)')
        plt.xlabel('Cell Area')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_area_distribution_v3.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return kept_data

    def compute_anomaly_scores(self, cell_images):
        """
        細胞画像のリストから異常関連スコアを計算する。
        v3: Latent Features (features_pca) も返す
        """
        if len(cell_images) == 0: return {}
        
        X = np.array(cell_images).astype('float32')
        
        batch_size = 32
        mse_list = []
        features_list = []
        residuals_list = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            reconstructed = self.autoencoder.predict(batch_X, verbose=0)
            diff = batch_X - reconstructed
            residuals_flat = diff.reshape(len(batch_X), -1)
            residuals_list.append(residuals_flat)
            batch_mse = np.mean(np.square(diff), axis=(1, 2, 3))
            mse_list.append(batch_mse)
            encoded_features = self.encoder.predict(batch_X, verbose=0)
            encoded_flat = encoded_features.reshape(len(encoded_features), -1)
            features_list.append(encoded_flat)
            
        final_mse = np.concatenate(mse_list, axis=0)
        final_features = np.concatenate(features_list, axis=0)
        final_residuals = np.concatenate(residuals_list, axis=0)
        
        encoded_scaled = self.scaler.transform(final_features)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        if self.detector_conservative:
            predictions = self.detector_conservative.predict(encoded_pca)
            anomaly_scores = -self.detector_conservative.decision_function(encoded_pca)
        else:
            anomaly_scores = final_mse 
            threshold = np.percentile(anomaly_scores, 95)
            predictions = np.where(anomaly_scores > threshold, -1, 1)

        return {
            'mse': final_mse, 
            'predictions': predictions, 
            'anomaly_scores': anomaly_scores,
            'anomaly_rate': np.sum(predictions == -1) / len(predictions),
            'features_pca': encoded_pca,
            'encoded_scaled': encoded_scaled,
            'residuals': final_residuals
        }

    def _get_files_from_paths(self, input_paths):
        files_dict = {}
        for path in input_paths:
            if os.path.isdir(path):
                tif_files = sorted(glob(os.path.join(path, '*.tif')) + glob(os.path.join(path, '*.tiff')))
                for f in tif_files:
                    files_dict[os.path.splitext(os.path.basename(f))[0]] = f
            elif os.path.isfile(path) and (path.endswith('.tif') or path.endswith('.tiff')):
                files_dict[os.path.splitext(os.path.basename(path))[0]] = path
        return files_dict

    def _get_folders_from_path(self, root_path):
        if not os.path.isdir(root_path): return {}
        target_folders = {}
        for root, dirs, files in os.walk(root_path):
            has_tif = any(f.endswith('.tif') or f.endswith('.tiff') for f in files)
            if has_tif:
                rel_path = os.path.relpath(root, root_path)
                if rel_path == '.':
                    sample_name = os.path.basename(root_path)
                else:
                    sample_name = os.path.basename(root)
                
                if sample_name in target_folders:
                    sample_name = f"{sample_name}_{os.path.basename(os.path.dirname(root))}"
                
                target_folders[sample_name] = root
        return target_folders

    def _calculate_wt_baseline_from_cells(self, wt_cells):
        if not wt_cells:
            return {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        
        wt_scores = self.compute_anomaly_scores(wt_cells)
        wt_rate = wt_scores['anomaly_rate'] * 100
        p99_score = np.quantile(wt_scores['anomaly_scores'], 0.99)
        # Threshold: Empirical value (wt_rate + margin) or purely statistical
        # For simplicity, keeping existing logic but annotating it.
        thresholds = {'wt_rate': wt_rate, 'threshold': wt_rate + 4.2, 'p99_score': p99_score}
        print(f"  WT Baseline Rate: {wt_rate:.2f}% | Threshold: {thresholds['threshold']:.2f}% | 99th Score: {p99_score:.2f}")
        return thresholds

    def run_folder_mode(self, root_path, output_dir, generate_umap, run_extra_viz, run_quantitative, wt_path=None):
        print(f"\n=== Running in FOLDER mode (v3: Robust & Latent Features) ===")
        folders_dict = self._get_folders_from_path(root_path)
        if not folders_dict:
            print("No subfolders found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        
        # --- 1. Load ALL Data first ---
        print("Loading all data from folders...")
        all_extracted_data = [] 
        
        for name, folder_path in folders_dict.items():
            print(f"  Loading folder: {name}...", end='\r')
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
            if not tif_files: continue
            
            for f_path in tif_files:
                cells_data = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw_c, pre_c, area, al_method) in enumerate(cells_data):
                    all_extracted_data.append({
                        'sample': name,
                        'folder_path': folder_path,
                        'file_path': f_path,
                        'local_idx': i,
                        'raw': raw_c,
                        'pre': pre_c,
                        'area': area,
                        'align_method': al_method
                    })
        print(f"\n  Loaded {len(all_extracted_data)} cells total.")

        # Load external WT if needed
        wt_sample_name_in_data = next((d['sample'] for d in all_extracted_data if 'WT' in d['sample'].upper() or d['sample'].upper() == 'WT'), None)
        
        if wt_path and not wt_sample_name_in_data:
            print(f"  WT not found in folders. Loading external WT from: {wt_path}")
            wt_files = []
            if os.path.isdir(wt_path):
                wt_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
            elif os.path.isfile(wt_path):
                wt_files = [wt_path]
            
            for f_path in wt_files:
                cells_data = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw_c, pre_c, area, al_method) in enumerate(cells_data):
                    all_extracted_data.append({
                        'sample': 'WT',
                        'folder_path': wt_path,
                        'file_path': f_path,
                        'local_idx': i,
                        'raw': raw_c,
                        'pre': pre_c,
                        'area': area,
                        'align_method': al_method
                    })

        # --- 2. Analyze Size & Soft Filter Garbage ---
        filtered_data = self.analyze_size_distribution(all_extracted_data, output_dir)
        
        # --- 3. Compute WT Baseline ---
        wt_data = [d['pre'] for d in filtered_data if 'WT' in d['sample'].upper() or d['sample'].upper() == 'WT']
        wt_thresholds = self._calculate_wt_baseline_from_cells(wt_data)
        
        # --- 4. Process Samples ---
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'residuals': [], 'sample_name': [], 'is_anomaly': [], 'mse': [], 'anomaly_score': []}
        
        unique_samples = sorted(list(set(d['sample'] for d in filtered_data)))
        
        for name in unique_samples:
            print(f"  Processing sample: {name}...")
            sample_entries = [d for d in filtered_data if d['sample'] == name]
            sample_cells = [d['pre'] for d in sample_entries]
            
            if not sample_cells: continue
            
            scores = self.compute_anomaly_scores(sample_cells)
            folder_path = sample_entries[0]['folder_path']
            
            summary_results[name] = {
                'sample_name': name, 
                'folder_path': folder_path, 
                'total_cells': len(sample_cells), 
                'anomaly_rate': scores['anomaly_rate'], 
                'mean_mse': np.mean(scores['mse']), 
                'is_wt': 'WT' in name.upper() or name.upper() == 'WT'
            }
            
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                entry = sample_entries[i]
                detailed_results.append({
                    'sample_name': name, 
                    'file_path': entry['file_path'], 
                    'local_idx': entry['local_idx'], 
                    'anomaly_score': score,
                    'mse': mse,
                    'area': entry['area'],
                    'size_z_score': entry.get('size_z_score', 0),
                    'align_method': entry['align_method']
                })
                
            if generate_umap or run_extra_viz or run_quantitative:
                # Use Latent Features (PCA) for clustering
                analysis_data['features'].append(scores['features_pca']) 
                # Keep residuals for comparison if needed, but primary is features
                analysis_data['residuals'].append(scores['residuals'])
                
                analysis_data['sample_name'].extend([name] * len(sample_cells))
                analysis_data['is_anomaly'].extend(scores['predictions'] == -1)
                analysis_data['mse'].extend(scores['mse'])
                analysis_data['anomaly_score'].extend(scores['anomaly_scores'])

        if not summary_results: 
            print("No results to save.")
            return

        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_folder_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_folder_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "Folder")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "Folder")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='folder')
        
        # --- Run XAI Analysis ---
        print("  Running XAI analysis...")
        wt_name = next((s for s in summary_results.keys() if 'WT' == s.upper() or 'WT' in s.upper()), None)
        if wt_name:
            wt_rows = df_detailed[df_detailed['sample_name'] == wt_name]
            if not wt_rows.empty:
                median_mse = wt_rows['mse'].median()
                wt_rows = wt_rows.copy()
                wt_rows['diff_from_median'] = (wt_rows['mse'] - median_mse).abs()
                median_candidates = wt_rows.sort_values('diff_from_median').head(5)
                
                for rank, (_, row) in enumerate(median_candidates.iterrows()):
                    try:
                        w_path = row['file_path']
                        w_idx = int(row['local_idx'])
                        w_data = self.extract_quality_cells(w_path, enhance_contrast=True)
                        if w_idx < len(w_data):
                            w_raw, w_pre, _, _ = w_data[w_idx] # Unpack
                            self.visualize_residuals(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_residuals.png'))
                            self.visualize_heatmap_overlay(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_heatmap.png'))
                    except Exception as e:
                        print(f"    [Warning] Failed to generate WT Reference {rank+1}: {e}")

        # Top 5 Candidates
        for sample in df_detailed['sample_name'].unique():
            s_rows = df_detailed[df_detailed['sample_name'] == sample]
            top_5 = s_rows.nlargest(5, 'anomaly_score')
            safe_sample_name = "".join(c for c in sample if c.isalnum() or c in ('-', '_')).rstrip()
            for rank, (_, row) in enumerate(top_5.iterrows()):
                try:
                    c_path = row['file_path']
                    c_idx = int(row['local_idx'])
                    c_data = self.extract_quality_cells(c_path, enhance_contrast=True)
                    if c_idx < len(c_data):
                        c_raw, c_pre, _, _ = c_data[c_idx]
                        base_name = f"{safe_sample_name}_rank{rank+1}_cell{c_idx}"
                        self.visualize_residuals(c_raw, c_pre, os.path.join(output_dir, f"xai_residuals_{base_name}.png"))
                        self.visualize_heatmap_overlay(c_raw, c_pre, os.path.join(output_dir, f"xai_heatmap_{base_name}.png"))
                except Exception as e:
                    print(f"    [Warning] Failed XAI for {sample} rank {rank+1}: {e}")

        # --- Run Extended Analyses ---
        if generate_umap or run_extra_viz or run_quantitative:
            if not analysis_data['features']:
                print("No features found for extended analysis.")
                return
            
            # v3: Use PCA Latent Features for Analysis
            all_features = np.concatenate(analysis_data['features'], axis=0)
            
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly'], 'anomaly_score': analysis_data['anomaly_score']})
            samples = sorted(analysis_df['sample'].unique())
            palette = sns.color_palette(self.okabe_ito, len(samples))
            color_map = dict(zip(samples, palette))

            if generate_umap:
                self.create_umap_visualization(all_features, analysis_df, output_dir, color_map)
            if run_extra_viz:
                self.create_pca_visualization(all_features, analysis_df, output_dir, color_map)
                self.create_tsne_visualization(all_features, analysis_df, output_dir, color_map)
                
            if generate_umap or run_extra_viz:
                self.create_wt_vs_mutant_visualizations(analysis_df, output_dir, color_map)

            if run_quantitative:
                self.calculate_distribution_distances(df_detailed, output_dir)
                self.perform_clustering_analysis(all_features, analysis_df, output_dir, color_map, df_detailed)
        
        print(f"Folder mode processing complete. Results are in {output_dir}")

    # --- File Mode Stub (Adapted to new return values) ---
    def run_file_mode(self, input_paths, output_dir, wt_path=None, generate_umap=False, run_extra_viz=False, run_quantitative=False):
        # Simplified copy of folder mode logic for file inputs
        print(f"\n=== Running in FILE mode (v3) ===")
        files_dict = self._get_files_from_paths(input_paths)
        if not files_dict:
            print("No TIF files found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        
        all_extracted_data = []
        for name, path in files_dict.items():
            cells_data = self.extract_quality_cells(path, enhance_contrast=True)
            for i, (raw_c, pre_c, area, al_method) in enumerate(cells_data):
                all_extracted_data.append({
                    'sample': name,
                    'file_path': path,
                    'local_idx': i, # Use local_idx as cell_id
                    'raw': raw_c,
                    'pre': pre_c,
                    'area': area,
                    'align_method': al_method
                })
        
        # Load external WT if needed
        wt_sample_name_in_data = next((d['sample'] for d in all_extracted_data if 'WT' in d['sample'].upper() or d['sample'].upper() == 'WT'), None)
        if wt_path and not wt_sample_name_in_data:
             wt_files = [wt_path] if os.path.isfile(wt_path) else sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
             for f_path in wt_files:
                cells_data = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw_c, pre_c, area, al_method) in enumerate(cells_data):
                    all_extracted_data.append({
                        'sample': 'WT',
                        'file_path': f_path,
                        'local_idx': i,
                        'raw': raw_c,
                        'pre': pre_c,
                        'area': area,
                        'align_method': al_method
                    })

        filtered_data = self.analyze_size_distribution(all_extracted_data, output_dir)
        wt_data = [d['pre'] for d in filtered_data if 'WT' in d['sample'].upper() or d['sample'].upper() == 'WT']
        wt_thresholds = self._calculate_wt_baseline_from_cells(wt_data)
        
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': [], 'anomaly_score': []}
        unique_samples = sorted(list(set(d['sample'] for d in filtered_data)))
        
        for name in unique_samples:
            sample_entries = [d for d in filtered_data if d['sample'] == name]
            sample_cells = [d['pre'] for d in sample_entries]
            if not sample_cells: continue
            
            scores = self.compute_anomaly_scores(sample_cells)
            summary_results[name] = {
                'sample_name': name, 
                'file_path': sample_entries[0]['file_path'], 
                'total_cells': len(sample_cells), 
                'anomaly_rate': scores['anomaly_rate'], 
                'mean_mse': np.mean(scores['mse']), 
                'is_wt': 'WT' in name.upper() or name.upper() == 'WT'
            }
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                entry = sample_entries[i]
                detailed_results.append({
                    'sample_name': name, 
                    'file_path': entry['file_path'], 
                    'cell_id': entry['local_idx'], 
                    'anomaly_score': score, 
                    'mse': mse,
                    'area': entry['area'],
                    'size_z_score': entry.get('size_z_score', 0)
                })
            
            if generate_umap or run_extra_viz or run_quantitative:
                analysis_data['features'].append(scores['features_pca'])
                analysis_data['sample_name'].extend([name] * len(sample_cells))
                analysis_data['is_anomaly'].extend(scores['predictions'] == -1)
                analysis_data['mse'].extend(scores['mse'])
                analysis_data['anomaly_score'].extend(scores['anomaly_scores'])

        if not summary_results: return

        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_file_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_file_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "File")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "File")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='file')
        
        # XAI & Extended Analysis skipped for brevity in file mode, but structure is ready.
        # Just running basic Extended Analysis if requested.
        if generate_umap or run_extra_viz or run_quantitative:
            all_features = np.concatenate(analysis_data['features'], axis=0)
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly'], 'anomaly_score': analysis_data['anomaly_score']})
            samples = sorted(analysis_df['sample'].unique())
            color_map = dict(zip(samples, sns.color_palette(self.okabe_ito, len(samples))))
            
            if generate_umap: self.create_umap_visualization(all_features, analysis_df, output_dir, color_map)
            if run_quantitative: self.perform_clustering_analysis(all_features, analysis_df, output_dir, color_map, df_detailed)

    # --- XAI Methods (Same as before) ---
    def visualize_2ch(self, image_data):
        if image_data.ndim == 2: return image_data 
        h, w, c = image_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if c >= 1: rgb[..., 0] = image_data[..., 0] # Red
        if c >= 2: rgb[..., 1] = image_data[..., 1] # Green
        return np.clip(rgb, 0, 1)

    def visualize_residuals(self, raw_image, preprocessed_image, save_path):
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)[0]
        if reconstructed.ndim == 2: reconstructed = np.expand_dims(reconstructed, axis=-1)
        if preprocessed_image.shape[-1] == 2 and reconstructed.shape[-1] == 1:
            reconstructed = np.concatenate([reconstructed, reconstructed], axis=-1)
        elif preprocessed_image.shape[-1] != reconstructed.shape[-1]:
            min_c = min(preprocessed_image.shape[-1], reconstructed.shape[-1])
            preprocessed_image_viz = preprocessed_image[..., :min_c]
            reconstructed_viz = reconstructed[..., :min_c]
        else:
            preprocessed_image_viz = preprocessed_image
            reconstructed_viz = reconstructed
        diff = np.mean(np.abs(preprocessed_image_viz - reconstructed_viz), axis=-1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(self.visualize_2ch(raw_image)); plt.axis('off')
        plt.subplot(1, 3, 2); plt.title("Reconstructed"); plt.imshow(self.visualize_2ch(reconstructed)); plt.axis('off')
        plt.subplot(1, 3, 3); plt.title("Residuals"); plt.imshow(diff, cmap='inferno'); plt.colorbar(); plt.axis('off')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def visualize_heatmap_overlay(self, raw_image, preprocessed_image, save_path):
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)[0]
        if reconstructed.ndim == 2: reconstructed = np.expand_dims(reconstructed, axis=-1)
        if preprocessed_image.shape[-1] == 2 and reconstructed.shape[-1] == 1:
            reconstructed = np.concatenate([reconstructed, reconstructed], axis=-1)
        elif preprocessed_image.shape[-1] != reconstructed.shape[-1]:
            min_c = min(preprocessed_image.shape[-1], reconstructed.shape[-1])
            preprocessed_image_viz = preprocessed_image[..., :min_c]
            reconstructed_viz = reconstructed[..., :min_c]
        else:
            preprocessed_image_viz = preprocessed_image
            reconstructed_viz = reconstructed
        diff = np.mean(np.abs(preprocessed_image_viz - reconstructed_viz), axis=-1)
        plt.figure(figsize=(6, 6))
        plt.imshow(self.visualize_2ch(raw_image))
        plt.imshow(diff, cmap='jet', alpha=0.5)
        plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    # --- Standard Viz ---
    def plot_anomaly_rates(self, df, output_dir, thresholds, mode_name):
        plt.figure(figsize=(14, 7))
        names = [n[:20] for n in df['sample_name']]
        colors = ['#333333' if is_wt else '#E69F00' for is_wt in df['is_wt']]
        plt.bar(range(len(names)), df['anomaly_rate'] * 100, color=colors, alpha=0.8)
        plt.axhline(thresholds['wt_rate'], color='#0072B2', linestyle='--')
        plt.axhline(thresholds['threshold'], color='#D55E00', linestyle='--')
        plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Anomaly Rate (%)'); plt.title(f'Anomaly Rates ({mode_name} Mode)'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_anomaly_rates_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    def plot_violin(self, df_detailed, output_dir, thresholds, mode_name):
        plt.figure(figsize=(14, 8))
        samples = df_detailed['sample_name'].unique()
        wt_name = next((s for s in samples if s.upper() == 'WT'), None)
        order = [wt_name] + sorted([s for s in samples if s.upper() != 'WT']) if wt_name else sorted(samples)
        sns.violinplot(x='sample_name', y='anomaly_score', hue='sample_name', legend=False, 
                       data=df_detailed, order=order, palette=self.okabe_ito, inner='quartile')
        plt.axhline(thresholds['p99_score'], color='#D55E00', linestyle='--')
        plt.xticks(rotation=45, ha='right'); plt.title(f'Anomaly Scores ({mode_name})'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_violin_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    def generate_phenotype_mosaic(self, df_summary, df_detailed, output_dir, thresholds, mode, top_n_samples=5, top_n_cells=5):
        mutants = df_summary[~df_summary['is_wt']].sort_values('anomaly_rate', ascending=False).head(top_n_samples)
        targets = mutants['sample_name'].tolist()
        if 'WT' in df_summary.index and 'WT' not in targets: targets.insert(0, 'WT')
        if not targets: return
        fig, axes = plt.subplots(len(targets), top_n_cells, figsize=(top_n_cells * 2, len(targets) * 2.2))
        if len(targets) == 1: axes = np.array([axes])
        for r, sample_name in enumerate(targets):
            s_data = df_detailed[df_detailed['sample_name'] == sample_name]
            if s_data.empty: continue
            label_suffix, candidates = ("(Typical)", s_data.nsmallest(top_n_cells, 'anomaly_score')) if 'WT' in sample_name.upper() else ("(Anomaly)", s_data.nlargest(top_n_cells, 'anomaly_score'))
            axes[r, 0].text(-0.2, 0.5, f"{sample_name}\n{label_suffix}", transform=axes[r, 0].transAxes, va='center', ha='right', fontsize=11, fontweight='bold')
            if mode == 'file':
                cells_data = self.extract_quality_cells(df_summary.loc[sample_name, 'file_path'], enhance_contrast=False)
                raw_cells = [c[0] for c in cells_data]
            for c, (_, cand_row) in enumerate(candidates.iterrows()):
                ax = axes[r, c]
                if mode == 'folder':
                    cells_data = self.extract_quality_cells(cand_row['file_path'], enhance_contrast=False)
                    raw_cells = [c[0] for c in cells_data]
                cell_idx = cand_row.get('local_idx', cand_row.get('cell_id'))
                if cell_idx < len(raw_cells):
                    img, score = raw_cells[cell_idx], cand_row['anomaly_score']
                    ax.imshow(self.visualize_2ch(img), vmin=0, vmax=1)
                    ax.set_title(f"{score:.1f}", color='red' if score > thresholds['p99_score'] else 'black', fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.suptitle("Phenotype Mosaic"); plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'plot_phenotype_mosaic_{mode.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    # --- Extended Viz (UMAP/PCA/tSNE/PHATE) ---
    def _plot_embedding(self, df, x_col, y_col, title, filename, color_map):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x_col, y=y_col, hue='sample', style='sample', size='is_anomaly', sizes=(10, 40), alpha=0.7, data=df, palette=color_map)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[s], markersize=10) for s in color_map]
        plt.legend(handles, color_map.keys(), title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

    def create_umap_visualization(self, features, df, output_dir, color_map):
        print("  Generating UMAP plot...")
        n_components = min(50, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(features)
        
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features_reduced)
        df['UMAP1'], df['UMAP2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'UMAP1', 'UMAP2', 'UMAP 2D Projection', os.path.join(output_dir, 'plot_umap.png'), color_map)

    def create_pca_visualization(self, features, df, output_dir, color_map):
        n_components = min(50, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        features_pca = pca.fit_transform(features)
        df['PCA1'], df['PCA2'] = features_pca[:, 0], features_pca[:, 1]
        self._plot_embedding(df, 'PCA1', 'PCA2', 'PCA Projection', os.path.join(output_dir, 'plot_pca.png'), color_map)

    def create_tsne_visualization(self, features, df, output_dir, color_map):
        if features.shape[1] > 50:
             pca = PCA(n_components=50, random_state=42)
             features = pca.fit_transform(features)
        embedding = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features)
        df['tSNE1'], df['tSNE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'tSNE1', 'tSNE2', 't-SNE Projection', os.path.join(output_dir, 'plot_tsne.png'), color_map)

    def create_wt_vs_mutant_visualizations(self, df, output_dir, color_map):
        # (Same as before, reusing existing plot_highlight logic internally or simplifying)
        pass # Skipped for brevity, existing logic assumed or user can rely on UMAP. 
        # Actually I should include it if user wants it. I'll include a simplified version.
        wt_sample = next((s for s in df['sample'].unique() if s.upper()=='WT'), None)
        if not wt_sample: return
        for mutant in [s for s in df['sample'].unique() if s != wt_sample]:
            if 'UMAP1' in df.columns:
                plt.figure(figsize=(10, 8))
                mask_wt, mask_mut = df['sample'] == wt_sample, df['sample'] == mutant
                plt.scatter(df.loc[~(mask_wt|mask_mut), 'UMAP1'], df.loc[~(mask_wt|mask_mut), 'UMAP2'], c='lightgray', alpha=0.1, s=10)
                plt.scatter(df.loc[mask_wt, 'UMAP1'], df.loc[mask_wt, 'UMAP2'], c='black', alpha=0.2, s=30)
                plt.scatter(df.loc[mask_mut, 'UMAP1'], df.loc[mask_mut, 'UMAP2'], c='red', alpha=0.6, s=40)
                plt.title(f"UMAP: WT vs {mutant}")
                plt.savefig(os.path.join(output_dir, f'umap_compare_WT_vs_{mutant}.png'), dpi=300, bbox_inches='tight'); plt.close()

    def perform_clustering_analysis(self, features, df, output_dir, color_map, df_detailed):
        # v3: Use Leiden (Scanpy) instead of K-Means
        if not SCANPY_AVAILABLE:
            print("  [Warning] Scanpy not found. Falling back to K-Means (k=10).")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(features)
        else:
            print("  Performing Leiden clustering (Scanpy)...")
            adata = anndata.AnnData(X=features)
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
            sc.tl.leiden(adata, resolution=0.5)
            df['cluster'] = adata.obs['leiden'].values

        # Sync & Plot
        if len(df) == len(df_detailed): df_detailed['cluster'] = df['cluster'].values
        
        # Composition Heatmap
        composition = df.groupby(['sample', 'cluster']).size().unstack(fill_value=0)
        composition_perc = composition.div(composition.sum(axis=1), axis=0) * 100
        composition_perc.to_csv(os.path.join(output_dir, 'quantitative_clustering_composition.csv'))
        
        wt_name = next((s for s in composition_perc.index if s.upper() == 'WT' or 'WT' in s.upper()), None)
        if wt_name:
            diff_df = composition_perc.subtract(composition_perc.loc[wt_name], axis=1)
            plt.figure(figsize=(12, len(diff_df)*0.5+2))
            sns.heatmap(diff_df, annot=True, fmt='.1f', cmap='RdBu_r', center=0, vmax=30, vmin=-30)
            plt.title('Cluster Diff (Sample - WT)'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quantitative_clustering_diff_heatmap.png'), dpi=300, bbox_inches='tight'); plt.close()

        self._generate_cluster_gallery(df_detailed, output_dir)

    def _generate_cluster_gallery(self, df_detailed, output_dir):
        if 'cluster' not in df_detailed.columns: return
        print("  Generating Cluster Cell Gallery...")
        unique_clusters = sorted(df_detailed['cluster'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
        for cluster_id in unique_clusters:
            cluster_cells = df_detailed[df_detailed['cluster'] == cluster_id]
            if cluster_cells.empty: continue
            sampled_cells = cluster_cells.sample(n=min(10, len(cluster_cells)), random_state=42)
            
            # Fetch images
            images = []
            for _, row in sampled_cells.iterrows():
                try:
                    c_data = self.extract_quality_cells(row['file_path'], enhance_contrast=False)
                    idx = int(row.get('local_idx', row.get('cell_id')))
                    if idx < len(c_data): images.append((c_data[idx][0], row['sample_name'])) # raw image
                except: pass
            
            if not images: continue
            fig, axes = plt.subplots(1, len(images), figsize=(len(images)*2, 3))
            if len(images)==1: axes=[axes]
            for i, ax in enumerate(axes):
                ax.imshow(self.visualize_2ch(images[i][0]))
                ax.set_title(images[i][1], fontsize=8); ax.axis('off')
            plt.suptitle(f"Cluster {cluster_id}"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_gallery.png"), dpi=300, bbox_inches='tight'); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Integrated Mutant Screening Pipeline (Aligned) v3 - Reviewer Response.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', required=True, choices=['file', 'folder', 'umap'])
    parser.add_argument('--input_paths', required=True, nargs='+')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output_dir', help="Output directory")
    parser.add_argument('--wt_path', help="WT path")
    parser.add_argument('--umap', action='store_true')
    parser.add_argument('--extra_viz', action='store_true')
    parser.add_argument('--quantitative', action='store_true')
    parser.add_argument('--use_prealigned', action='store_true')
    
    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"./screening_results/{timestamp}_{args.mode}_mode_aligned_v3"
    
    pipeline = MutantScreeningPipeline(args.model_dir, use_prealigned=args.use_prealigned)
    gen_umap = (args.mode == 'umap') or args.umap
    
    if args.mode == 'file':
        pipeline.run_file_mode(args.input_paths, output_dir, args.wt_path, gen_umap, args.extra_viz, args.quantitative)
    elif args.mode in ['folder', 'umap']:
        if len(args.input_paths) > 1: print("Warning: Using first input path only.")
        pipeline.run_folder_mode(args.input_paths[0], output_dir, gen_umap, args.extra_viz, args.quantitative, args.wt_path)

if __name__ == "__main__":
    main()
