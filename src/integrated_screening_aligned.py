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
    変異株スクリーニングの統合パイプライン (Aligned Model 対応版)。
    - ピレノイド（緑）を下向きに整列させて推論を行います。
    - ファイル単位での解析 (file mode)
    - フォルダ単位での解析 (folder mode)
    - UMAP可視化 (umap mode)
    - 追加の高度な解析 (extra_viz, quantitative flags)
    をサポートします。
    """

    def __init__(self, model_dir, use_prealigned=False):
        self.model_dir = model_dir
        self.use_prealigned = use_prealigned
        
        # --- Visual Style Setup (Okabe-Ito & Publication Ready) ---
        # Okabe-Ito Palette (Colorblind friendly)
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
            # 1. Find the last MaxPooling2D layer (output of encoder)
            maxpool_idx = -1
            for i, layer in enumerate(self.autoencoder.layers):
                if isinstance(layer, MaxPooling2D):
                    maxpool_idx = i
            
            if maxpool_idx != -1:
                # 2. Extract decoder layers (everything after the last MaxPooling2D)
                decoder_layers = self.autoencoder.layers[maxpool_idx+1:]
                
                # 3. Construct the decoder model
                # Input shape matches the encoder output (8, 8, 32)
                decoder_input = Input(shape=(8, 8, 32))
                x = decoder_input
                for layer in decoder_layers:
                    x = layer(x)
                self.decoder = Model(decoder_input, x)
                print("  Decoder extracted and reconstructed successfully.")
            else:
                pass

            self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'), compile=False)
            
            # --- Load Scaler (Priority: v2 -> v1) ---
            scaler_path_v2 = os.path.join(self.model_dir, 'scaler_v2.pkl')
            scaler_path_v1 = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path_v2):
                print(f"  Loading Scaler V2 from {scaler_path_v2}")
                with open(scaler_path_v2, 'rb') as f: self.scaler = pickle.load(f)
            else:
                print(f"  Loading Scaler V1 from {scaler_path_v1}")
                with open(scaler_path_v1, 'rb') as f: self.scaler = pickle.load(f)

            # --- Load PCA (Priority: v2 -> v1) ---
            pca_path_v2 = os.path.join(self.model_dir, 'pca_v2.pkl')
            pca_path_v1 = os.path.join(self.model_dir, 'pca.pkl')
            if os.path.exists(pca_path_v2):
                print(f"  Loading PCA V2 from {pca_path_v2}")
                with open(pca_path_v2, 'rb') as f: self.pca = pickle.load(f)
            else:
                print(f"  Loading PCA V1 from {pca_path_v1}")
                with open(pca_path_v1, 'rb') as f: self.pca = pickle.load(f)

            # --- Load Detector (Priority: svm_v2 -> conservative) ---
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
        """単一のTIF画像から品質基準を満たし、アライメント済みの細胞画像を抽出する"""
        try:
            if self.use_prealigned:
                try:
                    image = tiff.imread(image_path)
                    if image.shape == (64, 64, 2):
                        return [(image, image)]
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
                if minr < 5 or minc < 5 or maxr > H - 5 or maxc > W - 5:
                    continue

                # 2. Strict Shape Filtering (Quality Control)
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                if prop.solidity < 0.9: continue
                
                circularity = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                if circularity < 0.8: continue

                # 3. Centroid-based Padding & Crop
                cy, cx = map(int, prop.centroid)
                
                # Determine crop size (1.5x larger dimension, min 64)
                bbox_h = maxr - minr
                bbox_w = maxc - minc
                size = int(max(bbox_h, bbox_w) * 1.5)
                size = max(size, 64)
                half_size = size // 2
                
                # Calculate crop coordinates centered on centroid
                r_start = cy - half_size
                c_start = cx - half_size
                r_end = r_start + size
                c_end = c_start + size
                
                # Prepare padded containers
                crop_red = np.zeros((size, size), dtype=red_channel.dtype)
                crop_green = np.zeros((size, size), dtype=green_channel.dtype)
                crop_mask = np.zeros((size, size), dtype=bool)
                
                # Calculate overlap
                r_start_clamped = max(0, r_start)
                r_end_clamped = min(H, r_end)
                c_start_clamped = max(0, c_start)
                c_end_clamped = min(W, c_end)
                
                # Destination indices
                dr_start = r_start_clamped - r_start
                dr_end = dr_start + (r_end_clamped - r_start_clamped)
                dc_start = c_start_clamped - c_start
                dc_end = dc_start + (c_end_clamped - c_start_clamped)
                
                if dr_end > dr_start and dc_end > dc_start:
                    crop_red[dr_start:dr_end, dc_start:dc_end] = red_channel[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped]
                    crop_green[dr_start:dr_end, dc_start:dc_end] = green_channel[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped]
                    # Only mask the specific cell
                    mask_slice = (labels[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped] == prop.label)
                    crop_mask[dr_start:dr_end, dc_start:dc_end] = mask_slice
                else:
                    continue

                # Apply Mask
                crop_red = crop_red * crop_mask
                crop_green = crop_green * crop_mask

                # 4. Alignment Logic
                # Recalculate Centroid of Cell (Red) within crop
                try:
                    m = regionprops(crop_mask.astype(int), intensity_image=crop_red)[0]
                    cy_crop, cx_crop = m.centroid
                except IndexError:
                    continue
                
                if np.max(crop_green) < p99_green * 0.1: continue

                # Calculate Centroid of Pyrenoid (Green)
                try:
                    mg = regionprops(crop_mask.astype(int), intensity_image=crop_green)[0]
                    py, px = mg.weighted_centroid
                except IndexError:
                    continue

                dy, dx = py - cy_crop, px - cx_crop
                
                if dy**2 + dx**2 < 2.0:
                    angle = 0
                else:
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    angle = angle_deg - 90

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
                
                quality_cells.append((aligned_cell, aligned_cell))
                
            return quality_cells
        except Exception as e:
            print(f"Error extracting cells from {os.path.basename(image_path)}: {e}")
            return []

    def compute_anomaly_scores(self, cell_images):
        """
        細胞画像のリストから異常関連スコアを計算する。
        Test-Time Augmentation (TTA) は削除しました（アライメント済みのため）。
        """
        if len(cell_images) == 0: return {}
        
        # Images are already (N, 64, 64, 2)
        X = np.array(cell_images).astype('float32')
        
        # 1. Reconstruction & MSE
        # Batch processing to avoid OOM
        batch_size = 32
        mse_list = []
        features_list = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            
            # Reconstruction
            reconstructed = self.autoencoder.predict(batch_X, verbose=0)
            batch_mse = np.mean(np.square(batch_X - reconstructed), axis=(1, 2, 3))
            mse_list.append(batch_mse)
            
            # Encoder features
            encoded_features = self.encoder.predict(batch_X, verbose=0)
            encoded_flat = encoded_features.reshape(len(encoded_features), -1)
            features_list.append(encoded_flat)
            
        final_mse = np.concatenate(mse_list, axis=0) # (N,)
        final_features = np.concatenate(features_list, axis=0) # (N, feature_dim)
        
        # 2. Standard PCA & Anomaly Detection pipeline
        encoded_scaled = self.scaler.transform(final_features)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        if self.detector_conservative:
            predictions = self.detector_conservative.predict(encoded_pca)
            anomaly_scores = -self.detector_conservative.decision_function(encoded_pca)
        else:
            # Fallback if no detector loaded: use MSE as score
            # Normalize MSE to behave somewhat like decision function for compatibility
            anomaly_scores = final_mse 
            threshold = np.percentile(anomaly_scores, 95)
            predictions = np.where(anomaly_scores > threshold, -1, 1)

        return {
            'mse': final_mse, 
            'predictions': predictions, 
            'anomaly_scores': anomaly_scores,
            'anomaly_rate': np.sum(predictions == -1) / len(predictions),
            'features_pca': encoded_pca
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
            # Check if this folder contains any tif files
            has_tif = any(f.endswith('.tif') or f.endswith('.tiff') for f in files)
            if has_tif:
                # Use the relative path from root_path as the sample name, replacing separators with underscores
                rel_path = os.path.relpath(root, root_path)
                if rel_path == '.':
                    sample_name = os.path.basename(root_path)
                else:
                    # e.g. Mutant/mutantA -> Mutant_mutantA
                    # or just use the folder name if unique? Let's use full relative path to be safe.
                    # sample_name = rel_path.replace(os.sep, '_')
                    # Actually, user might prefer just the folder name if unique. 
                    # Let's stick to folder name for simplicity, but handle duplicates?
                    # "Mutant/mutantA" -> "mutantA"
                    sample_name = os.path.basename(root)
                
                # Check for duplicates
                if sample_name in target_folders:
                    # Fallback to unique name
                    sample_name = f"{sample_name}_{os.path.basename(os.path.dirname(root))}"
                
                target_folders[sample_name] = root
        return target_folders

    def _calculate_wt_baseline(self, wt_path):
        print(f"Calculating baseline from WT: {wt_path}")
        wt_cells = []
        if os.path.isdir(wt_path):
            tif_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
            for f in tif_files:
                # Unpack tuples and keep only preprocessed cells for scoring
                cells_data = self.extract_quality_cells(f, enhance_contrast=True)
                wt_cells.extend([c[1] for c in cells_data])
        elif os.path.isfile(wt_path):
            cells_data = self.extract_quality_cells(wt_path, enhance_contrast=True)
            wt_cells = [c[1] for c in cells_data]
        if not wt_cells:
            print("Warning: No cells found in WT. Using default threshold.")
            return {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        wt_scores = self.compute_anomaly_scores(wt_cells)
        wt_rate = wt_scores['anomaly_rate'] * 100
        p99_score = np.quantile(wt_scores['anomaly_scores'], 0.99)
        thresholds = {'wt_rate': wt_rate, 'threshold': wt_rate + 4.2, 'p99_score': p99_score}
        print(f"  WT Baseline Rate: {wt_rate:.2f}% | Threshold: {thresholds['threshold']:.2f}% | 99th Score: {p99_score:.2f}")
        return thresholds

    def run_file_mode(self, input_paths, output_dir, wt_path=None, generate_umap=False, run_extra_viz=False, run_quantitative=False):
        print(f"\n=== Running in FILE mode (UMAP: {generate_umap}, ExtraViz: {run_extra_viz}, Quantitative: {run_quantitative}) ===")
        files_dict = self._get_files_from_paths(input_paths)
        if not files_dict:
            print("No TIF files found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        wt_thresholds = self._calculate_wt_baseline(wt_path) if wt_path else {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': []}
        
        for name, path in files_dict.items():
            print(f"  Processing {name}...")
            cells_data = self.extract_quality_cells(path, enhance_contrast=True)
            if not cells_data: continue
            
            # Unpack: we use preprocessed for scoring
            preprocessed_cells = [c[1] for c in cells_data]
            
            scores = self.compute_anomaly_scores(preprocessed_cells)
            summary_results[name] = {'sample_name': name, 'file_path': path, 'total_cells': len(preprocessed_cells), 'anomaly_rate': scores['anomaly_rate'], 'mean_mse': np.mean(scores['mse']), 'is_wt': 'WT' in name.upper()}
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                detailed_results.append({'sample_name': name, 'file_path': path, 'cell_id': i, 'anomaly_score': score, 'mse': mse})
            
            if generate_umap or run_extra_viz or run_quantitative:
                analysis_data['features'].append(scores['features_pca'])
                analysis_data['sample_name'].extend([name] * len(preprocessed_cells))
                analysis_data['is_anomaly'].extend(scores['predictions'] == -1)
                analysis_data['mse'].extend(scores['mse'])

        if not summary_results:
            print("No results to save.")
            return
            
        # --- Ensure WT is included in analysis_data if available ---
        if (generate_umap or run_extra_viz or run_quantitative) and wt_path:
            # Check if WT was already processed from the input files
            wt_in_files = any(name.upper() == 'WT' or 'WT' in name.upper() for name in files_dict.keys())
            
            should_load_wt = False
            if not wt_in_files:
                should_load_wt = True
                print(f"  WT not found in input files. Loading additional WT data from: {wt_path}")
            else:
                 pass

            if should_load_wt:
                wt_files = []
                if os.path.isdir(wt_path):
                    wt_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
                elif os.path.isfile(wt_path):
                    wt_files = [wt_path]
                
                wt_cells_accum = []
                wt_cell_metadata = [] 
                for wf in wt_files:
                    w_data = self.extract_quality_cells(wf, enhance_contrast=True)
                    for i, (raw_c, pre_c) in enumerate(w_data):
                        wt_cells_accum.append(pre_c)
                        wt_cell_metadata.append((wf, i))
                
                if wt_cells_accum:
                    print(f"    Loaded {len(wt_cells_accum)} cells from wt_path.")
                    wt_scores = self.compute_anomaly_scores(wt_cells_accum)
                    
                    analysis_data['features'].append(wt_scores['features_pca'])
                    analysis_data['sample_name'].extend(['WT'] * len(wt_cells_accum))
                    analysis_data['is_anomaly'].extend(wt_scores['predictions'] == -1)
                    analysis_data['mse'].extend(wt_scores['mse'])
                    
                    if 'WT' not in summary_results:
                         summary_results['WT'] = {
                            'sample_name': 'WT', 
                            'file_path': wt_path, 
                            'total_cells': len(wt_cells_accum), 
                            'anomaly_rate': wt_scores['anomaly_rate'], 
                            'mean_mse': np.mean(wt_scores['mse']), 
                            'is_wt': True
                        }
                         for i, (score, mse) in enumerate(zip(wt_scores['anomaly_scores'], wt_scores['mse'])):
                            f_path, local_idx = wt_cell_metadata[i]
                            detailed_results.append({
                                'sample_name': 'WT', 
                                'file_path': f_path, 
                                'cell_id': i, 
                                'anomaly_score': score, 
                                'mse': mse
                            })

        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_file_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_file_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "File")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "File")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='file')
        
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
                        w_idx = int(row['cell_id'])
                        w_data = self.extract_quality_cells(w_path, enhance_contrast=True)
                        if w_idx < len(w_data):
                            w_raw, w_pre = w_data[w_idx]
                            self.visualize_residuals(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_residuals.png'))
                            self.visualize_heatmap_overlay(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_heatmap.png'))
                    except Exception as e:
                         print(f"    [Warning] Failed to generate WT Reference {rank+1}: {e}")

        # 2. Top 5 Candidates per File
        for sample in df_detailed['sample_name'].unique():
            s_rows = df_detailed[df_detailed['sample_name'] == sample]
            top_5 = s_rows.nlargest(5, 'anomaly_score')
            safe_sample_name = "".join(c for c in sample if c.isalnum() or c in ('-', '_')).rstrip()
            for rank, (_, row) in enumerate(top_5.iterrows()):
                try:
                    c_path = row['file_path']
                    c_idx = int(row['cell_id'])
                    c_data = self.extract_quality_cells(c_path, enhance_contrast=True)
                    if c_idx < len(c_data):
                        c_raw, c_pre = c_data[c_idx]
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
            
            all_features_pca = np.concatenate(analysis_data['features'], axis=0)
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly']})
            
            samples = sorted(analysis_df['sample'].unique())
            # Use Okabe-Ito (cycling if needed)
            palette = sns.color_palette(self.okabe_ito, len(samples))
            color_map = dict(zip(samples, palette))

            if generate_umap:
                self.create_umap_visualization(all_features_pca, analysis_df, output_dir, color_map)
            if run_extra_viz:
                self.create_pca_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_tsne_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_phate_visualization(all_features_pca, analysis_df, output_dir, color_map)
            
            if generate_umap or run_extra_viz:
                self.create_wt_vs_mutant_visualizations(analysis_df, output_dir, color_map)

            if run_quantitative:
                self.calculate_distribution_distances(df_detailed, output_dir)
                self.perform_clustering_analysis(all_features_pca, analysis_df, output_dir, color_map, df_detailed)

        print(f"File mode processing complete. Results are in {output_dir}")

    def run_folder_mode(self, root_path, output_dir, generate_umap, run_extra_viz, run_quantitative, wt_path=None):
        print(f"\n=== Running in FOLDER mode (UMAP: {generate_umap}, ExtraViz: {run_extra_viz}, Quantitative: {run_quantitative}) ===")
        folders_dict = self._get_folders_from_path(root_path)
        if not folders_dict:
            print("No subfolders found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        if not wt_path and 'WT' in folders_dict:
            wt_path = folders_dict['WT']
        wt_thresholds = self._calculate_wt_baseline(wt_path) if wt_path else {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': []}
        for name, folder_path in folders_dict.items():
            print(f"  Processing folder: {name}...")
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
            if not tif_files: continue
            folder_cells, cell_metadata = [], []
            for f_path in tif_files:
                # Unpack tuple: (raw, preprocessed)
                cells_data = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw_c, pre_c) in enumerate(cells_data):
                    folder_cells.append(pre_c)
                    cell_metadata.append((f_path, i))
            if not folder_cells: continue
            scores = self.compute_anomaly_scores(folder_cells)
            summary_results[name] = {'sample_name': name, 'folder_path': folder_path, 'total_cells': len(folder_cells), 'anomaly_rate': scores['anomaly_rate'], 'mean_mse': np.mean(scores['mse']), 'is_wt': 'WT' in name.upper()}
            for i, (score, mse) in enumerate(zip(scores['anomaly_scores'], scores['mse'])):
                f_path, local_idx = cell_metadata[i]
                detailed_results.append({'sample_name': name, 'file_path': f_path, 'local_idx': local_idx, 'anomaly_score': score, 'mse': mse})
            if generate_umap or run_extra_viz or run_quantitative:
                analysis_data['features'].append(scores['features_pca'])
                analysis_data['sample_name'].extend([name] * len(folder_cells))
                analysis_data['is_anomaly'].extend(scores['predictions'] == -1)
        if not summary_results:
            print("No results to save.")
            return

        # --- Ensure WT is included in analysis_data if available ---
        if (generate_umap or run_extra_viz or run_quantitative) and wt_path:
            wt_in_folders = False
            wt_folder_path = None
            for name, path in folders_dict.items():
                if name.upper() == 'WT':
                    wt_in_folders = True
                    wt_folder_path = path
                    break
            
            should_load_wt = False
            if not wt_in_folders:
                should_load_wt = True
                print(f"  WT not found in folders. Loading additional WT data from: {wt_path}")
            else:
                try:
                    if os.path.abspath(wt_path) != os.path.abspath(wt_folder_path):
                        should_load_wt = True
                        print(f"  WT found in folders, but --wt_path is different. Merging data from: {wt_path}")
                    else:
                        print(f"  WT found in folders and matches --wt_path. Using loaded data.")
                except Exception:
                    pass

            if should_load_wt:
                wt_files = []
                if os.path.isdir(wt_path):
                    wt_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
                elif os.path.isfile(wt_path):
                    wt_files = [wt_path]
                
                wt_cells_accum = []
                wt_cell_metadata = [] 
                for wf in wt_files:
                    w_data = self.extract_quality_cells(wf, enhance_contrast=True)
                    for i, (raw_c, pre_c) in enumerate(w_data):
                        wt_cells_accum.append(pre_c)
                        wt_cell_metadata.append((wf, i))
                
                if wt_cells_accum:
                    print(f"    Loaded {len(wt_cells_accum)} cells from wt_path.")
                    wt_scores = self.compute_anomaly_scores(wt_cells_accum)
                    
                    analysis_data['features'].append(wt_scores['features_pca'])
                    analysis_data['sample_name'].extend(['WT'] * len(wt_cells_accum))
                    analysis_data['is_anomaly'].extend(wt_scores['predictions'] == -1)
                    analysis_data['mse'].extend(wt_scores['mse'])
                    
                    summary_results['WT'] = {
                        'sample_name': 'WT', 
                        'folder_path': wt_path, 
                        'total_cells': len(wt_cells_accum), 
                        'anomaly_rate': wt_scores['anomaly_rate'], 
                        'mean_mse': np.mean(wt_scores['mse']), 
                        'is_wt': True
                    }
                    
                    for i, (score, mse) in enumerate(zip(wt_scores['anomaly_scores'], wt_scores['mse'])):
                        f_path, local_idx = wt_cell_metadata[i]
                        detailed_results.append({
                            'sample_name': 'WT', 
                            'file_path': f_path, 
                            'local_idx': local_idx, 
                            'anomaly_score': score, 
                            'mse': mse
                        })

        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, 'summary_folder_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_folder_mode.csv'), index=False)
        self.plot_anomaly_rates(df_summary, output_dir, wt_thresholds, "Folder")
        self.plot_violin(df_detailed, output_dir, wt_thresholds, "Folder")
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, wt_thresholds, mode='folder')
        
        # --- Run XAI Analysis (New Logic) ---
        print("  Running XAI analysis (WT Reference 5 cells & Top 5 Candidates)...")
        
        wt_name = next((s for s in summary_results.keys() if 'WT' == s.upper() or 'WT' in s.upper()), None)
        if 'WT' in summary_results:
             wt_name = 'WT'

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
                        w_idx = int(row['local_idx']) if 'local_idx' in row else int(row['cell_id'])
                        w_data = self.extract_quality_cells(w_path, enhance_contrast=True)
                        if w_idx < len(w_data):
                            w_raw, w_pre = w_data[w_idx]
                            self.visualize_residuals(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_residuals.png'))
                            self.visualize_heatmap_overlay(w_raw, w_pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}_heatmap.png'))
                    except Exception as e:
                        print(f"    [Warning] Failed to generate WT Reference {rank+1}: {e}")

        # 2. Top 5 Candidates per Series
        for sample in df_detailed['sample_name'].unique():
            s_rows = df_detailed[df_detailed['sample_name'] == sample]
            top_5 = s_rows.nlargest(5, 'anomaly_score')
            safe_sample_name = "".join(c for c in sample if c.isalnum() or c in ('-', '_')).rstrip()
            for rank, (_, row) in enumerate(top_5.iterrows()):
                try:
                    c_path = row['file_path']
                    c_idx = int(row['local_idx']) if 'local_idx' in row else int(row['cell_id'])
                    c_data = self.extract_quality_cells(c_path, enhance_contrast=True)
                    
                    if c_idx < len(c_data):
                        c_raw, c_pre = c_data[c_idx]
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
            
            all_features_pca = np.concatenate(analysis_data['features'], axis=0)
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly']})
            
            samples = sorted(analysis_df['sample'].unique())
            # Use Okabe-Ito (cycling if needed)
            palette = sns.color_palette(self.okabe_ito, len(samples))
            color_map = dict(zip(samples, palette))

            if generate_umap:
                self.create_umap_visualization(all_features_pca, analysis_df, output_dir, color_map)
            if run_extra_viz:
                self.create_pca_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_tsne_visualization(all_features_pca, analysis_df, output_dir, color_map)
                self.create_phate_visualization(all_features_pca, analysis_df, output_dir, color_map)
            
            if generate_umap or run_extra_viz:
                self.create_wt_vs_mutant_visualizations(analysis_df, output_dir, color_map)

            if run_quantitative:
                self.calculate_distribution_distances(df_detailed, output_dir)
                self.perform_clustering_analysis(all_features_pca, analysis_df, output_dir, color_map, df_detailed)
        
        print(f"Folder mode processing complete. Results are in {output_dir}")

    # --- XAI Methods ---
    def visualize_2ch(self, image_data):
        """ (64, 64, 2) -> (64, 64, 3) RGB Conversion """
        if image_data.ndim == 2: return image_data # Already grayscale
        h, w, c = image_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if c >= 1: rgb[..., 0] = image_data[..., 0] # Red (Chl)
        if c >= 2: rgb[..., 1] = image_data[..., 1] # Green (Pyr)
        return np.clip(rgb, 0, 1)

    def visualize_residuals(self, raw_image, preprocessed_image, save_path):
        """
        Original (Raw), Reconstructed (from Preprocessed), and Difference Heatmap.
        """
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        
        # Predict returns (1, 64, 64, C) or (1, 64, 64)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)[0]
        
        # 1. Dimension Check: Ensure (64, 64, C)
        if reconstructed.ndim == 2:
            reconstructed = np.expand_dims(reconstructed, axis=-1)
            
        # 2. Channel Mismatch Check (Input 2ch vs Output 1ch)
        if preprocessed_image.shape[-1] == 2 and reconstructed.shape[-1] == 1:
            reconstructed = np.concatenate([reconstructed, reconstructed], axis=-1)
        elif preprocessed_image.shape[-1] != reconstructed.shape[-1]:
            # Fallback: slice to minimum common channels
            min_c = min(preprocessed_image.shape[-1], reconstructed.shape[-1])
            preprocessed_image_viz = preprocessed_image[..., :min_c]
            reconstructed_viz = reconstructed[..., :min_c]
        else:
            preprocessed_image_viz = preprocessed_image
            reconstructed_viz = reconstructed

        diff = np.mean(np.abs(preprocessed_image_viz - reconstructed_viz), axis=-1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title("Original (Raw)")
        plt.imshow(self.visualize_2ch(raw_image))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Reconstructed")
        plt.imshow(self.visualize_2ch(reconstructed))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Difference Heatmap")
        plt.imshow(diff, cmap='inferno')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_heatmap_overlay(self, raw_image, preprocessed_image, save_path):
        """
        Overlay reconstruction error heatmap on the raw image.
        """
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)[0]
        
        if reconstructed.ndim == 2:
            reconstructed = np.expand_dims(reconstructed, axis=-1)
            
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
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # --- Standard Visualization Methods ---
    def plot_anomaly_rates(self, df, output_dir, thresholds, mode_name):
        plt.figure(figsize=(14, 7))
        names = [n[:20] for n in df['sample_name']]
        # Contrast: WT=DarkGray, Others=Orange
        colors = ['#333333' if is_wt else '#E69F00' for is_wt in df['is_wt']]
        plt.bar(range(len(names)), df['anomaly_rate'] * 100, color=colors, alpha=0.8)
        plt.axhline(thresholds['wt_rate'], color='#0072B2', linestyle='--', label=f"WT Baseline ({thresholds['wt_rate']:.1f}%)") # Blue
        plt.axhline(thresholds['threshold'], color='#D55E00', linestyle='--', label=f"Hit Threshold ({thresholds['threshold']:.1f}%)") # Vermilion
        plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Anomaly Rate (%)'); plt.title(f'Anomaly Rates by Sample ({mode_name} Mode)'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_anomaly_rates_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    def plot_violin(self, df_detailed, output_dir, thresholds, mode_name):
        print("  Generating Violin Plot...")
        samples = df_detailed['sample_name'].unique()
        wt_name = next((s for s in samples if s.upper() == 'WT'), None)

        plt.figure(figsize=(14, 8))
        order = [wt_name] + sorted([s for s in samples if s.upper() != 'WT']) if wt_name else sorted(samples)
        
        # Updated for modern Seaborn: x, y, hue (same as x), legend=False
        sns.violinplot(x='sample_name', y='anomaly_score', hue='sample_name', legend=False, 
                       data=df_detailed, order=order, palette=self.okabe_ito, inner='quartile')
        
        plt.axhline(thresholds['p99_score'], color='#D55E00', linestyle='--', linewidth=2, label=f'WT 99th Percentile ({thresholds["p99_score"]:.2f})')
        plt.legend(loc='upper right'); plt.title(f'Anomaly Score Distribution ({mode_name} Mode)'); plt.ylabel('Anomaly Score (Higher = More Abnormal)'); plt.xlabel('Sample')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_violin_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

        if wt_name and mode_name.lower() == 'folder':
            mutants = [s for s in samples if s != wt_name]
            for mutant in mutants:
                plt.figure(figsize=(8, 6))
                sub_df = df_detailed[df_detailed['sample_name'].isin([wt_name, mutant])]
                # Contrast: WT=Black, Mutant=Vermilion
                sns.violinplot(x='sample_name', y='anomaly_score', hue='sample_name', legend=False,
                               data=sub_df, order=[wt_name, mutant], palette=['#333333', '#D55E00'])
                
                plt.axhline(thresholds['p99_score'], color='#0072B2', linestyle='--', linewidth=2, label=f'WT 99th Percentile ({thresholds["p99_score"]:.2f})')
                plt.legend(loc='upper right')
                plt.title(f'Anomaly Score: WT vs {mutant}')
                plt.ylabel('Anomaly Score')
                plt.xlabel('')
                plt.tight_layout()
                sanitized_mutant_name = "".join(c for c in mutant if c.isalnum() or c in ('-', '_')).rstrip()
                plt.savefig(os.path.join(output_dir, f'plot_violin_WT_vs_{sanitized_mutant_name}_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight')
                plt.close()

    def generate_phenotype_mosaic(self, df_summary, df_detailed, output_dir, thresholds, mode, top_n_samples=5, top_n_cells=5):
        print("  Generating Phenotype Mosaic...")
        mutants = df_summary[~df_summary['is_wt']].sort_values('anomaly_rate', ascending=False).head(top_n_samples)
        targets = mutants['sample_name'].tolist()
        if 'WT' in df_summary.index and 'WT' not in targets:
            targets.insert(0, 'WT')
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
        plt.suptitle("Phenotype Mosaic (Raw Images)"); plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'plot_phenotype_mosaic_{mode.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    # --- Extended Visualization & Analysis Methods ---
    def _plot_embedding(self, df, x_col, y_col, title, filename, color_map):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x_col, y=y_col, hue='sample', style='sample', size='is_anomaly', sizes=(10, 40), alpha=0.7, data=df, palette=color_map)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[s], markersize=10) for s in color_map]
        plt.legend(handles, color_map.keys(), title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title); plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

    def create_umap_visualization(self, features, df, output_dir, color_map):
        print("  Generating UMAP plot...")
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        df['UMAP1'], df['UMAP2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'UMAP1', 'UMAP2', 'UMAP 2D Projection', os.path.join(output_dir, 'plot_umap.png'), color_map)

        # --- MSE Visualization on UMAP ---
        if 'mse' in df.columns:
            print("  Generating UMAP MSE plot...")
            # Sort by MSE so high MSE points are plotted on top
            df_sorted = df.sort_values('mse', ascending=True)
            
            plt.figure(figsize=(10, 8))
            # cmap='cividis' for colorblind friendly sequential, sort ascending
            sc = plt.scatter(df_sorted['UMAP1'], df_sorted['UMAP2'], c=df_sorted['mse'], cmap='cividis', s=10, alpha=0.8)
            plt.colorbar(sc, label='Reconstruction MSE')
            plt.title('UMAP Colored by Reconstruction Error (MSE)')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plot_umap_mse.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def create_pca_visualization(self, features, df, output_dir, color_map):
        print("  Generating PCA plot...")
        df['PCA1'], df['PCA2'] = features[:, 0], features[:, 1]
        self._plot_embedding(df, 'PCA1', 'PCA2', 'PCA Projection (First 2 Components)', os.path.join(output_dir, 'plot_pca.png'), color_map)

    def create_tsne_visualization(self, features, df, output_dir, color_map):
        print("  Generating t-SNE plot...")
        embedding = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300).fit_transform(features)
        df['tSNE1'], df['tSNE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'tSNE1', 'tSNE2', 't-SNE 2D Projection', os.path.join(output_dir, 'plot_tsne.png'), color_map)

    def create_phate_visualization(self, features, df, output_dir, color_map):
        if phate is None:
            print("  Skipping PHATE plot: 'phate' library not installed.")
            return
        print("  Generating PHATE plot...")
        phate_op = phate.PHATE(random_state=42)
        embedding = phate_op.fit_transform(features)
        df['PHATE1'], df['PHATE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'PHATE1', 'PHATE2', 'PHATE 2D Projection', os.path.join(output_dir, 'plot_phate.png'), color_map)

    def create_wt_vs_mutant_visualizations(self, df, output_dir, color_map):
        print("  Generating WT vs Mutant comparison plots...")
        wt_sample_name = next((s for s in df['sample'].unique() if s.upper() == 'WT'), None)
        
        if not wt_sample_name:
            print("  Skipping WT vs Mutant plots: WT sample not found.")
            return

        mutant_samples = [s for s in df['sample'].unique() if s != wt_sample_name]
        
        for mutant in mutant_samples:
            sanitized_mutant_name = "".join(c for c in mutant if c.isalnum() or c in ('-', '_')).rstrip()
            
            def plot_highlight(x_col, y_col, plot_name, filename):
                plt.figure(figsize=(10, 8))
                
                mask_wt = df['sample'] == wt_sample_name
                mask_mutant = df['sample'] == mutant
                mask_others = ~(mask_wt | mask_mutant)
                
                # Others: Gray, small, transparent
                plt.scatter(df.loc[mask_others, x_col], df.loc[mask_others, y_col], 
                            c='lightgray', alpha=0.2, s=15, label='Others', edgecolors='none', marker='.')
                
                # WT: Black/DarkGray, Circle, more distinct
                plt.scatter(df.loc[mask_wt, x_col], df.loc[mask_wt, y_col], 
                            c='#000000', alpha=0.6, s=30, label=wt_sample_name, edgecolors='none', marker='o')
                
                # Mutant: Vermilion (from Okabe-Ito), Triangle, distinct
                plt.scatter(df.loc[mask_mutant, x_col], df.loc[mask_mutant, y_col], 
                            c='#D55E00', alpha=0.8, s=40, label=mutant, edgecolors='white', linewidth=0.5, marker='^')
                
                plt.title(f"{plot_name}: WT vs {mutant}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

            if 'UMAP1' in df.columns:
                plot_highlight('UMAP1', 'UMAP2', 'UMAP', os.path.join(output_dir, f'umap_compare_WT_vs_{sanitized_mutant_name}.png'))

    def calculate_distribution_distances(self, df_detailed, output_dir):
        print("  Calculating distribution distances from WT...")
        wt_scores = df_detailed[df_detailed['sample_name'].str.upper() == 'WT']['anomaly_score']
        if wt_scores.empty:
            print("  Skipping distribution distances: WT sample not found.")
            return
        distances = {}
        for sample in df_detailed['sample_name'].unique():
            if sample.upper() == 'WT': continue
            sample_scores = df_detailed[df_detailed['sample_name'] == sample]['anomaly_score']
            distances[sample] = wasserstein_distance(wt_scores, sample_scores)
        df_dist = pd.DataFrame.from_dict(distances, orient='index', columns=['wasserstein_distance_from_WT']).sort_values(by='wasserstein_distance_from_WT', ascending=False)
        df_dist.to_csv(os.path.join(output_dir, 'quantitative_distribution_distances.csv'))
        print(f"  Saved distribution distances to {os.path.join(output_dir, 'quantitative_distribution_distances.csv')}")

    def perform_clustering_analysis(self, features, df, output_dir, color_map, df_detailed):
        if not SCANPY_AVAILABLE:
            print("  Skipping Clustering: 'scanpy' or 'anndata' library not installed.")
            return
        
        print("  Performing Leiden clustering (Scanpy)...")
        # 1. Create AnnData object
        adata = anndata.AnnData(X=features)
        
        # 2. Compute neighbors
        # Use PCA features directly as 'X' (or we could store in obsm but using X is simpler here)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
        
        # 3. Leiden clustering
        resolution = 0.5
        sc.tl.leiden(adata, resolution=resolution)
        
        # 4. Store cluster labels back to df
        # Convert to int to match previous logic or keep as category string
        df['cluster'] = adata.obs['leiden'].values
        
        # Sync cluster labels to df_detailed (assuming same order)
        if len(df) == len(df_detailed):
             df_detailed['cluster'] = df['cluster'].values
        else:
             print("Warning: df and df_detailed length mismatch. Skipping cluster sync.")

        # --- Visualization ---
        
        # Composition analysis
        composition = df.groupby(['sample', 'cluster']).size().unstack(fill_value=0)
        composition_perc = composition.div(composition.sum(axis=1), axis=0) * 100
        composition_perc.to_csv(os.path.join(output_dir, 'quantitative_clustering_composition.csv'))
        
        # Plot composition heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(composition_perc, annot=True, fmt='.1f', cmap='viridis')
        plt.title(f'Cluster Composition (%) by Sample (Leiden res={resolution})')
        plt.xlabel('Cluster ID')
        plt.ylabel('Sample')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantitative_clustering_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot difference from WT heatmap
        wt_name = next((s for s in composition_perc.index if s.upper() == 'WT'), None)
        if wt_name:
            wt_comp = composition_perc.loc[wt_name]
            diff_df = composition_perc.subtract(wt_comp, axis=1)
            
            plt.figure(figsize=(12, 8))
            # cmap='PuOr' for colorblind friendly divergent (Purple-Orange), center=0
            sns.heatmap(diff_df, annot=True, fmt='.1f', cmap='PuOr', center=0)
            plt.title('Cluster Composition Difference (Sample - WT)')
            plt.xlabel('Cluster ID')
            plt.ylabel('Sample')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quantitative_clustering_diff_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Plot UMAP colored by cluster if UMAP data exists in df
        if 'UMAP1' in df.columns:
            plt.figure(figsize=(10, 8))
            # Ensure clusters are categorical for proper coloring
            unique_clusters = sorted(df['cluster'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
            num_clusters = len(unique_clusters)
            
            # Use tab10 or Okabe-Ito if small number, else tab20
            if num_clusters <= 10:
                palette = sns.color_palette("tab10", num_clusters)
            else:
                palette = sns.color_palette("tab20", num_clusters)
            
            sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', style='sample', data=df, palette=palette, alpha=0.7)
            plt.title('UMAP Projection with Leiden Clusters')
            plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plot_umap_leiden_clusters.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Option: Use Scanpy's UMAP plotting if we calculate UMAP within Scanpy too
            # But here we already have UMAP coords in df. We can inject them into adata.obsm['X_umap']
            adata.obsm['X_umap'] = df[['UMAP1', 'UMAP2']].values
            # Create Scanpy UMAP plot
            try:
                # Scanpy settings for saving
                sc.settings.figdir = output_dir
                sc.set_figure_params(dpi=300, facecolor='white')
                
                sc.pl.umap(adata, color='leiden', show=False, title='Leiden Clusters (Scanpy Style)', save='_scanpy_leiden.png')
                # Rename or move if needed, scanpy saves as "umap_scanpy_leiden.png" usually
                # We'll just let it save as configured.
                print(f"  Saved Scanpy-style UMAP plot to {output_dir}")
            except Exception as e:
                print(f"  Could not generate Scanpy UMAP plot: {e}")

        # WT cluster distribution bar plot
        if wt_name:
            wt_composition = composition_perc.loc[wt_name]
            plt.figure(figsize=(8, 6))
            wt_composition.plot(kind='bar', color='#56B4E9') # Sky Blue
            plt.title('Cluster Distribution for WT')
            plt.ylabel('Percentage (%)')
            plt.xlabel('Cluster ID')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quantitative_wt_cluster_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"  Saved clustering results to {output_dir}")
        
        # Generate Cluster Gallery
        self._generate_cluster_gallery(df_detailed, output_dir)

    def _generate_cluster_gallery(self, df_detailed, output_dir):
        """
        Generate a gallery of random cells (including mutants) for each cluster.
        """
        if 'cluster' not in df_detailed.columns:
            print("  Skipping cluster gallery: 'cluster' column not found in detailed results.")
            return

        print("  Generating Cluster Cell Gallery...")
        
        # Use ALL samples, not just WT
        all_df = df_detailed
        
        if all_df.empty:
            print("  Skipping cluster gallery: No cells found.")
            return

        unique_clusters = sorted(all_df['cluster'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
        
        for cluster_id in unique_clusters:
            cluster_cells = all_df[all_df['cluster'] == cluster_id]
            if cluster_cells.empty:
                continue
            
            # Sample up to 10 cells
            n_samples = min(10, len(cluster_cells))
            sampled_cells = cluster_cells.sample(n=n_samples, random_state=42)
            
            # Collect images.
            file_groups = sampled_cells.groupby('file_path')
            
            # Temporary storage: {index_in_sampled -> image_data}
            temp_images = {}
            
            for f_path, group in file_groups:
                try:
                    # Note: We read the file again.
                    # Since we want visual quality, we set enhance_contrast=False to see Raw/Aligned data
                    # Or we can set True if we prefer enhanced.
                    # Prompt says "visualize_residuals ... use preprocessed_image".
                    # Let's use Raw (aligned but not enhanced) for gallery to see true morphology.
                    # Actually, our extract_quality_cells returns (aligned, aligned) now.
                    
                    needed_indices = []
                    for idx, row in group.iterrows():
                        cell_idx = int(row.get('local_idx', row.get('cell_id')))
                        needed_indices.append((idx, cell_idx)) 
                    
                    cells_data = self.extract_quality_cells(f_path, enhance_contrast=False)
                    
                    for df_idx, c_idx in needed_indices:
                        if c_idx < len(cells_data):
                            raw_img = cells_data[c_idx][0]
                            temp_images[df_idx] = raw_img
                        else:
                            print(f"Warning: Cell index {c_idx} out of bounds for {f_path}")
                            
                except Exception as e:
                    print(f"Error reading file for gallery {f_path}: {e}")

            # Reconstruct list
            final_images = []
            final_labels = []
            
            for idx, row in sampled_cells.iterrows():
                if idx in temp_images:
                    final_images.append(temp_images[idx])
                    # Label: SampleName: Score
                    final_labels.append(f"{row['sample_name']}\n{row['anomaly_score']:.1f}")

            if not final_images:
                continue

            # Plot
            n_cols = len(final_images)
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 3))
            if n_cols == 1: axes = [axes]
            
            for i, ax in enumerate(axes):
                ax.imshow(self.visualize_2ch(final_images[i])) 
                # Usually gallery shows structure. Let's show Green (pyrenoid) since alignment is based on it.
                # Or better, show RGB if possible, or just Green.
                # Let's show Green channel as grayscale for clarity of pyrenoid shape.
                
                ax.axis('off')
                ax.set_title(final_labels[i], fontsize=8)
            
            plt.suptitle(f"Cluster {cluster_id} (n={len(cluster_cells)})", fontsize=14)
            plt.tight_layout()
            save_name = f"cluster_{cluster_id}_gallery.png"
            plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Integrated Mutant Screening Pipeline (Aligned).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', required=True, choices=['file', 'folder', 'umap'], help="Analysis mode:\n'file':   Process each .tif file in INPUT_PATHS as a separate sample.\n'folder': Process each subfolder in INPUT_PATHS as a sample.\n'umap':   Same as 'folder' mode, but also generates UMAP plots.")
    parser.add_argument('--input_paths', required=True, nargs='+', help="One or more input files or directories.")
    parser.add_argument('--model_dir', required=True, help="Directory containing the trained models.")
    parser.add_argument('--output_dir', help="Directory to save results. If not given, a timestamped folder will be created.")
    parser.add_argument('--wt_path', help="Path to the WT file or folder for baseline calculation. If not provided, will look for 'WT' in inputs.")
    # --- New arguments for extended analysis ---
    parser.add_argument('--umap', action='store_true', help="Force UMAP generation (useful for 'file' or 'folder' modes).")
    parser.add_argument('--extra_viz', action='store_true', help="Generate extra visualizations (t-SNE, PCA, PHATE) in file/folder/umap mode.")
    parser.add_argument('--quantitative', action='store_true', help="Perform quantitative analysis (distribution distance, clustering) in file/folder/umap mode.")
    parser.add_argument('--use_prealigned', action='store_true', help="Use pre-aligned/cropped images (skip StarDist and rotation).")
    
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"./screening_results/{timestamp}_{args.mode}_mode_aligned"
    
    try:
        pipeline = MutantScreeningPipeline(args.model_dir, use_prealigned=args.use_prealigned)
        generate_umap = (args.mode == 'umap') or args.umap
        
        if args.mode == 'file':
            # Support extra args in file mode now
            pipeline.run_file_mode(args.input_paths, output_dir, args.wt_path, generate_umap, args.extra_viz, args.quantitative)
        elif args.mode in ['folder', 'umap']:
            if len(args.input_paths) > 1:
                print("Warning: In 'folder' or 'umap' mode, only the first input path is used as the root directory.")
            root_path = args.input_paths[0]
            pipeline.run_folder_mode(root_path, output_dir, generate_umap, args.extra_viz, args.quantitative, args.wt_path)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
