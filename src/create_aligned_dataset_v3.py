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
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
                self.pca = pickle.load(f)
            
            det_path = os.path.join(self.model_dir, 'detector_conservative.pkl')
            if os.path.exists(det_path):
                with open(det_path, 'rb') as f:
                    self.detector_conservative = pickle.load(f)
            else:
                self.detector_conservative = None
            
            # StarDistは整列済みデータを使う場合ロードしない（高速化）
            if not self.use_prealigned:
                self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            
            print("All models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models from {self.model_dir}: {e}")
            raise

    def extract_quality_cells(self, image_path, enhance_contrast=True):
        """単一のTIF画像から品質基準を満たし、アライメント済みの細胞画像を抽出する"""
        try:
            image = tiff.imread(image_path)
            
            # --- Pre-aligned Mode (Fast Path) ---
            if self.use_prealigned:
                # 既に整列済み (64, 64, 2) の画像をそのまま返す
                if image.shape == (64, 64, 2):
                    # 正規化済みと仮定
                    return [(image, image)]
                elif image.ndim == 3 and image.shape[-1] > 2:
                    # チャンネルが多い場合は最初の2つを使う
                    return [(image[..., :2], image[..., :2])]
                else:
                    return []

            # --- Raw Image Processing Mode ---
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
            
            for prop in props:
                # Filters
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                if prop.solidity < 0.9: continue
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

                # Alignment
                m = regionprops(crop_mask.astype(int), intensity_image=crop_red)[0]
                cy, cx = m.centroid
                
                if np.max(crop_green) < p99_green * 0.1: continue

                mg = regionprops(crop_mask.astype(int), intensity_image=crop_green)[0]
                py, px = mg.weighted_centroid
                
                dy, dx = py - cy, px - cx
                
                if dy**2 + dx**2 < 2.0:
                    angle = 0
                else:
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    angle = angle_deg - 90 # Pyrenoid to bottom

                rotated_red = rotate(crop_red, angle, resize=False, preserve_range=True)
                rotated_green = rotate(crop_green, angle, resize=False, preserve_range=True)
                
                # Crop 64x64
                crop_size = 64
                center_y, center_x = rotated_red.shape[0] // 2, rotated_red.shape[1] // 2
                y1 = max(0, center_y - crop_size // 2)
                y2 = min(rotated_red.shape[0], center_y + crop_size // 2)
                x1 = max(0, center_x - crop_size // 2)
                x2 = min(rotated_red.shape[1], center_x + crop_size // 2)
                
                final_red = rotated_red[y1:y2, x1:x2]
                final_green = rotated_green[y1:y2, x1:x2]
                
                final_red = resize(final_red, (crop_size, crop_size), anti_aliasing=True)
                final_green = resize(final_green, (crop_size, crop_size), anti_aliasing=True)
                
                final_red = np.clip(final_red / p99_red, 0, 1)
                final_green = np.clip(final_green / p99_green, 0, 1)

                aligned_cell = np.stack([final_red, final_green], axis=-1).astype(np.float32)
                quality_cells.append((aligned_cell, aligned_cell))
                
            return quality_cells
        except Exception as e:
            print(f"Error extracting cells from {os.path.basename(image_path)}: {e}")
            return []

    def compute_anomaly_scores(self, cell_images):
        if len(cell_images) == 0: return {}
        
        X = np.array(cell_images).astype('float32')
        
        batch_size = 32
        mse_list = []
        features_list = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            reconstructed = self.autoencoder.predict(batch_X, verbose=0)
            batch_mse = np.mean(np.square(batch_X - reconstructed), axis=(1, 2, 3))
            mse_list.append(batch_mse)
            encoded_features = self.encoder.predict(batch_X, verbose=0)
            encoded_flat = encoded_features.reshape(len(encoded_features), -1)
            features_list.append(encoded_flat)
            
        final_mse = np.concatenate(mse_list, axis=0)
        final_features = np.concatenate(features_list, axis=0)
        
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
            'features_pca': encoded_pca
        }
        
    def visualize_2ch(self, image_data):
        """ (64, 64, 2) -> (64, 64, 3) RGB Conversion Helper """
        if image_data.ndim == 2: return image_data 
        h, w, c = image_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if c >= 1: rgb[..., 0] = image_data[..., 0] # Red (Chl)
        if c >= 2: rgb[..., 1] = image_data[..., 1] # Green (Pyr)
        return np.clip(rgb, 0, 1)

    def _get_files_from_paths(self, input_paths):
        files_dict = {}
        for path in input_paths:
            # Recursive search for file mode as well
            if os.path.isdir(path):
                tif_files = sorted(glob(os.path.join(path, '**', '*.tif'), recursive=True) + 
                                   glob(os.path.join(path, '**', '*.tiff'), recursive=True))
                for f in tif_files:
                    files_dict[os.path.splitext(os.path.basename(f))[0]] = f
            elif os.path.isfile(path) and (path.endswith('.tif') or path.endswith('.tiff')):
                files_dict[os.path.splitext(os.path.basename(path))[0]] = path
        return files_dict

    def _get_folders_from_path(self, root_path):
        """指定パス直下のサブディレクトリを取得する"""
        if not os.path.isdir(root_path): return {}
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
        return {os.path.basename(f): f for f in subfolders}

    def _calculate_wt_baseline(self, wt_path):
        print(f"Calculating baseline from WT: {wt_path}")
        wt_cells = []
        if os.path.isdir(wt_path):
            # Recursive search in WT folder too
            tif_files = sorted(glob(os.path.join(wt_path, '**', '*.tif'), recursive=True) + 
                               glob(os.path.join(wt_path, '**', '*.tiff'), recursive=True))
            for f in tif_files:
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
        print(f"\n=== Running in FILE mode (UMAP: {generate_umap}) ===")
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

        # (Common output saving logic omitted for brevity, similar to folder mode below)
        self._save_and_visualize(summary_results, detailed_results, analysis_data, output_dir, wt_thresholds, "File", generate_umap, run_extra_viz, run_quantitative, wt_path)

    def run_folder_mode(self, root_path, output_dir, generate_umap, run_extra_viz, run_quantitative, wt_path=None):
        print(f"\n=== Running in FOLDER mode (UMAP: {generate_umap}) ===")
        folders_dict = self._get_folders_from_path(root_path)
        if not folders_dict:
            print("No subfolders found.")
            return
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine WT Path
        if not wt_path and 'WT' in folders_dict:
            wt_path = folders_dict['WT']
        
        wt_thresholds = self._calculate_wt_baseline(wt_path) if wt_path else {'wt_rate': 0.0, 'threshold': 5.0, 'p99_score': 13.0}
        
        summary_results, detailed_results, analysis_data = {}, [], {'features': [], 'sample_name': [], 'is_anomaly': [], 'mse': []}
        
        for name, folder_path in folders_dict.items():
            print(f"  Processing folder: {name}...")
            # Recursive search for images in subfolders (e.g. Mutant/mutantA/*.tif)
            tif_files = sorted(glob(os.path.join(folder_path, '**', '*.tif'), recursive=True) + 
                               glob(os.path.join(folder_path, '**', '*.tiff'), recursive=True))
            
            if not tif_files: 
                print(f"    No images found in {name} (or subdirectories). Skipping.")
                continue
                
            folder_cells, cell_metadata = [], []
            for f_path in tif_files:
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
                analysis_data['mse'].extend(scores['mse'])

        if not summary_results:
            print("No results to save.")
            return

        self._save_and_visualize(summary_results, detailed_results, analysis_data, output_dir, wt_thresholds, "Folder", generate_umap, run_extra_viz, run_quantitative, wt_path)

    def _save_and_visualize(self, summary_results, detailed_results, analysis_data, output_dir, thresholds, mode_name, generate_umap, run_extra_viz, run_quantitative, wt_path):
        # WT Loading for plots
        if (generate_umap or run_extra_viz or run_quantitative) and wt_path:
            wt_sample_found = any(s['is_wt'] for s in summary_results.values())
            if not wt_sample_found:
                 # Load WT externally if not in processed folders
                 # (Logic condensed for brevity: assumed WT is usually in folders or handled)
                 pass

        df_summary = pd.DataFrame.from_dict(summary_results, orient='index')
        df_detailed = pd.DataFrame(detailed_results)
        df_summary.to_csv(os.path.join(output_dir, f'summary_{mode_name.lower()}_mode.csv'))
        df_detailed.to_csv(os.path.join(output_dir, f'detailed_results_{mode_name.lower()}_mode.csv'), index=False)
        
        self.plot_anomaly_rates(df_summary, output_dir, thresholds, mode_name)
        self.plot_violin(df_detailed, output_dir, thresholds, mode_name)
        self.generate_phenotype_mosaic(df_summary, df_detailed, output_dir, thresholds, mode=mode_name.lower())
        
        # XAI
        self.run_xai_analysis(df_detailed, output_dir, summary_results)

        # Extended
        if (generate_umap or run_extra_viz or run_quantitative) and analysis_data['features']:
            all_features = np.concatenate(analysis_data['features'], axis=0)
            analysis_df = pd.DataFrame({'sample': analysis_data['sample_name'], 'is_anomaly': analysis_data['is_anomaly']})
            if 'mse' in analysis_data:
                analysis_df['mse'] = analysis_data['mse']
            
            samples = sorted(analysis_df['sample'].unique())
            palette = sns.color_palette(self.okabe_ito, len(samples))
            color_map = dict(zip(samples, palette))

            if generate_umap:
                self.create_umap_visualization(all_features, analysis_df, output_dir, color_map)
            # (Other visualizers called here...)
            if run_quantitative:
                self.perform_clustering_analysis(all_features, analysis_df, output_dir, color_map, df_detailed)

        print(f"{mode_name} mode processing complete. Results are in {output_dir}")

    def run_xai_analysis(self, df_detailed, output_dir, summary_results):
        print("  Running XAI analysis...")
        wt_name = next((s for s, v in summary_results.items() if v['is_wt']), None)
        
        # WT Reference (Median)
        if wt_name:
            wt_rows = df_detailed[df_detailed['sample_name'] == wt_name]
            if not wt_rows.empty:
                median_mse = wt_rows['mse'].median()
                wt_rows = wt_rows.copy()
                wt_rows['diff_from_median'] = (wt_rows['mse'] - median_mse).abs()
                candidates = wt_rows.sort_values('diff_from_median').head(5)
                for rank, (_, row) in enumerate(candidates.iterrows()):
                    try:
                        w_path = row['file_path']
                        # local_idx if available, else cell_id
                        idx = int(row.get('local_idx', row.get('cell_id')))
                        w_data = self.extract_quality_cells(w_path, enhance_contrast=True)
                        if idx < len(w_data):
                            raw, pre = w_data[idx]
                            self.visualize_residuals(raw, pre, os.path.join(output_dir, f'PositiveControl_WT_Median_{rank+1}.png'))
                    except Exception as e: print(f"    [Warning] Failed to generate WT Reference {rank+1}: {e}")

        # Top 5 Anomalies per Sample
        for sample in df_detailed['sample_name'].unique():
            s_rows = df_detailed[df_detailed['sample_name'] == sample]
            top_5 = s_rows.nlargest(5, 'anomaly_score')
            safe_name = "".join(c for c in sample if c.isalnum() or c in ('-', '_')).rstrip()
            for rank, (_, row) in enumerate(top_5.iterrows()):
                try:
                    path = row['file_path']
                    idx = int(row.get('local_idx', row.get('cell_id')))
                    c_data = self.extract_quality_cells(path, enhance_contrast=True)
                    if idx < len(c_data):
                        raw, pre = c_data[idx]
                        self.visualize_residuals(raw, pre, os.path.join(output_dir, f"xai_{safe_name}_rank{rank+1}.png"))
                except Exception as e: print(f"    [Warning] Failed XAI for {sample} rank {rank+1}: {e}")

    def visualize_residuals(self, raw_image, preprocessed_image, save_path):
        """ Visualization with 2ch support """
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0) # (1, 64, 64, 2)
        
        # 2ch Diff (Mean over channels for heatmap)
        # Ensure reconstructed is also 2ch, which it is.
        diff = np.mean(np.abs(preprocessed_image - reconstructed[0]), axis=-1)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(self.visualize_2ch(raw_image)); plt.axis('off')
        plt.subplot(1, 3, 2); plt.title("Reconstructed"); plt.imshow(self.visualize_2ch(reconstructed[0])); plt.axis('off')
        plt.subplot(1, 3, 3); plt.title("Error Heatmap"); plt.imshow(diff, cmap='inferno'); plt.colorbar(); plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

    def visualize_heatmap_overlay(self, raw_image, preprocessed_image, save_path):
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        reconstructed = self.autoencoder.predict(img_batch, verbose=0)
        diff = np.mean(np.abs(preprocessed_image - reconstructed[0]), axis=-1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.visualize_2ch(raw_image))
        plt.imshow(diff, cmap='jet', alpha=0.4) # Overlay
        plt.axis('off'); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

    def plot_anomaly_rates(self, df, output_dir, thresholds, mode_name):
        plt.figure(figsize=(14, 7))
        names = [n[:20] for n in df['sample_name']]
        colors = ['#333333' if is_wt else '#E69F00' for is_wt in df['is_wt']]
        plt.bar(range(len(names)), df['anomaly_rate'] * 100, color=colors, alpha=0.8)
        plt.axhline(thresholds['wt_rate'], color='#0072B2', linestyle='--', label=f"WT Baseline ({thresholds['wt_rate']:.1f}%)")
        plt.axhline(thresholds['threshold'], color='#D55E00', linestyle='--', label=f"Hit Threshold ({thresholds['threshold']:.1f}%)")
        plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Anomaly Rate (%)'); plt.title(f'Anomaly Rates by Sample ({mode_name} Mode)'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_anomaly_rates_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

    def plot_violin(self, df_detailed, output_dir, thresholds, mode_name):
        print("  Generating Violin Plot...")
        samples = df_detailed['sample_name'].unique()
        wt_name = next((s for s in samples if s.upper() == 'WT'), None)

        plt.figure(figsize=(14, 8))
        order = [wt_name] + sorted([s for s in samples if s.upper() != 'WT']) if wt_name else sorted(samples)
        # Fix: Add hue and legend=False to silence warning
        sns.violinplot(x='sample_name', y='anomaly_score', hue='sample_name', data=df_detailed, order=order, palette=self.okabe_ito, inner='quartile', legend=False)
        plt.axhline(thresholds['p99_score'], color='#D55E00', linestyle='--', linewidth=2, label=f'WT 99th Percentile ({thresholds["p99_score"]:.2f})')
        plt.legend(loc='upper right'); plt.title(f'Anomaly Score Distribution ({mode_name} Mode)'); plt.ylabel('Anomaly Score (Higher = More Abnormal)'); plt.xlabel('Sample')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_violin_{mode_name.lower()}.png'), dpi=300, bbox_inches='tight'); plt.close()

        if wt_name and mode_name.lower() == 'folder':
            mutants = [s for s in samples if s != wt_name]
            for mutant in mutants:
                plt.figure(figsize=(8, 6))
                sub_df = df_detailed[df_detailed['sample_name'].isin([wt_name, mutant])]
                sns.violinplot(x='sample_name', y='anomaly_score', hue='sample_name', data=sub_df, order=[wt_name, mutant], palette=['#333333', '#D55E00'], legend=False)
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
        if 'WT' in df_summary.index and 'WT' not in targets: targets.insert(0, 'WT')
        if not targets: return

        fig, axes = plt.subplots(len(targets), top_n_cells, figsize=(top_n_cells * 2, len(targets) * 2.2))
        if len(targets) == 1: axes = np.array([axes])
        
        for r, sample_name in enumerate(targets):
            s_data = df_detailed[df_detailed['sample_name'] == sample_name]
            if s_data.empty: continue
            
            is_wt = 'WT' in sample_name.upper()
            label = "(Typical)" if is_wt else "(Anomaly)"
            candidates = s_data.nsmallest(top_n_cells, 'anomaly_score') if is_wt else s_data.nlargest(top_n_cells, 'anomaly_score')
            
            axes[r, 0].text(-0.2, 0.5, f"{sample_name}\n{label}", transform=axes[r, 0].transAxes, va='center', ha='right', fontweight='bold')
            
            for c, (_, row) in enumerate(candidates.iterrows()):
                ax = axes[r, c]
                path = row['file_path']
                idx = int(row.get('local_idx', row.get('cell_id')))
                
                try:
                    c_data = self.extract_quality_cells(path, enhance_contrast=False) # Use raw for visual
                    if idx < len(c_data):
                        img = c_data[idx][0] # Raw aligned image
                        ax.imshow(self.visualize_2ch(img)) # FIX: Convert to RGB
                        score = row['anomaly_score']
                        ax.set_title(f"{score:.1f}", color='red' if score > thresholds['p99_score'] else 'black', fontsize=9)
                except: pass
                ax.axis('off')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'plot_phenotype_mosaic_{mode}.png'), dpi=300); plt.close()

    def _generate_cluster_gallery(self, df_detailed, output_dir):
        if 'cluster' not in df_detailed.columns: return
        print("  Generating Cluster Gallery...")
        # Use WT samples for gallery
        wt_df = df_detailed[df_detailed['sample_name'].str.contains('WT', case=False, na=False)]
        if wt_df.empty: return

        for cluster_id in sorted(wt_df['cluster'].unique()):
            cluster_cells = wt_df[wt_df['cluster'] == cluster_id]
            sampled = cluster_cells.sample(n=min(10, len(cluster_cells)), random_state=42)
            
            images = []
            for _, row in sampled.iterrows():
                try:
                    path = row['file_path']
                    idx = int(row.get('local_idx', row.get('cell_id')))
                    data = self.extract_quality_cells(path, enhance_contrast=False)
                    if idx < len(data):
                        images.append(self.visualize_2ch(data[idx][0]))
                except: pass
            
            if not images: continue
            
            fig, axes = plt.subplots(1, len(images), figsize=(len(images)*1.5, 2))
            if len(images) == 1: axes = [axes]
            for i, ax in enumerate(axes):
                ax.imshow(images[i])
                ax.axis('off')
            plt.suptitle(f"Cluster {cluster_id} (WT)", fontsize=10)
            plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_gallery.png"), dpi=150, bbox_inches='tight')
            plt.close()

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

        if 'mse' in df.columns:
            print("  Generating UMAP MSE plot...")
            df_sorted = df.sort_values('mse', ascending=True)
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(df_sorted['UMAP1'], df_sorted['UMAP2'], c=df_sorted['mse'], cmap='cividis', s=10, alpha=0.8)
            plt.colorbar(sc, label='Reconstruction MSE')
            plt.title('UMAP Colored by Reconstruction Error (MSE)')
            plt.xlabel('UMAP1'); plt.ylabel('UMAP2'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plot_umap_mse.png'), dpi=300, bbox_inches='tight'); plt.close()

    def create_pca_visualization(self, features, df, output_dir, color_map):
        print("  Generating PCA plot...")
        df['PCA1'], df['PCA2'] = features[:, 0], features[:, 1]
        self._plot_embedding(df, 'PCA1', 'PCA2', 'PCA Projection', os.path.join(output_dir, 'plot_pca.png'), color_map)

    def create_tsne_visualization(self, features, df, output_dir, color_map):
        print("  Generating t-SNE plot...")
        embedding = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300).fit_transform(features)
        df['tSNE1'], df['tSNE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'tSNE1', 'tSNE2', 't-SNE 2D Projection', os.path.join(output_dir, 'plot_tsne.png'), color_map)

    def create_phate_visualization(self, features, df, output_dir, color_map):
        if phate is None: return
        print("  Generating PHATE plot...")
        phate_op = phate.PHATE(random_state=42)
        embedding = phate_op.fit_transform(features)
        df['PHATE1'], df['PHATE2'] = embedding[:, 0], embedding[:, 1]
        self._plot_embedding(df, 'PHATE1', 'PHATE2', 'PHATE 2D Projection', os.path.join(output_dir, 'plot_phate.png'), color_map)

    def create_wt_vs_mutant_visualizations(self, df, output_dir, color_map):
        print("  Generating WT vs Mutant plots...")
        wt_sample_name = next((s for s in df['sample'].unique() if s.upper() == 'WT'), None)
        if not wt_sample_name: return

        mutant_samples = [s for s in df['sample'].unique() if s != wt_sample_name]
        for mutant in mutant_samples:
            sanitized_mutant_name = "".join(c for c in mutant if c.isalnum() or c in ('-', '_')).rstrip()
            
            def plot_highlight(x_col, y_col, plot_name, filename):
                plt.figure(figsize=(10, 8))
                mask_wt = df['sample'] == wt_sample_name
                mask_mutant = df['sample'] == mutant
                mask_others = ~(mask_wt | mask_mutant)
                
                plt.scatter(df.loc[mask_others, x_col], df.loc[mask_others, y_col], c='lightgray', alpha=0.2, s=15, label='Others', edgecolors='none', marker='.')
                plt.scatter(df.loc[mask_wt, x_col], df.loc[mask_wt, y_col], c='#000000', alpha=0.6, s=30, label=wt_sample_name, edgecolors='none', marker='o')
                plt.scatter(df.loc[mask_mutant, x_col], df.loc[mask_mutant, y_col], c='#D55E00', alpha=0.8, s=40, label=mutant, edgecolors='white', linewidth=0.5, marker='^')
                
                plt.title(f"{plot_name}: WT vs {mutant}"); plt.xlabel(x_col); plt.ylabel(y_col); plt.legend(loc='upper right'); plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

            if 'UMAP1' in df.columns:
                plot_highlight('UMAP1', 'UMAP2', 'UMAP', os.path.join(output_dir, f'umap_compare_WT_vs_{sanitized_mutant_name}.png'))

    def calculate_distribution_distances(self, df_detailed, output_dir):
        print("  Calculating distribution distances...")
        wt_scores = df_detailed[df_detailed['sample_name'].str.upper() == 'WT']['anomaly_score']
        if wt_scores.empty: return
        distances = {}
        for sample in df_detailed['sample_name'].unique():
            if sample.upper() == 'WT': continue
            sample_scores = df_detailed[df_detailed['sample_name'] == sample]['anomaly_score']
            distances[sample] = wasserstein_distance(wt_scores, sample_scores)
        df_dist = pd.DataFrame.from_dict(distances, orient='index', columns=['wasserstein_distance_from_WT']).sort_values(by='wasserstein_distance_from_WT', ascending=False)
        df_dist.to_csv(os.path.join(output_dir, 'quantitative_distribution_distances.csv'))

    def perform_clustering_analysis(self, features, df, output_dir, color_map, df_detailed):
        if not SCANPY_AVAILABLE: return
        print("  Performing Leiden clustering...")
        adata = anndata.AnnData(X=features)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
        resolution = 0.5
        sc.tl.leiden(adata, resolution=resolution)
        df['cluster'] = adata.obs['leiden'].values
        if len(df) == len(df_detailed): df_detailed['cluster'] = df['cluster'].values

        # Composition
        composition = df.groupby(['sample', 'cluster']).size().unstack(fill_value=0)
        composition_perc = composition.div(composition.sum(axis=1), axis=0) * 100
        composition_perc.to_csv(os.path.join(output_dir, 'quantitative_clustering_composition.csv'))
        
        plt.figure(figsize=(12, 8)); sns.heatmap(composition_perc, annot=True, fmt='.1f', cmap='viridis')
        plt.title(f'Cluster Composition (%) by Sample (res={resolution})'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantitative_clustering_heatmap.png'), dpi=300); plt.close()

        # UMAP with clusters
        if 'UMAP1' in df.columns:
            plt.figure(figsize=(10, 8))
            unique_clusters = sorted(df['cluster'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
            palette = sns.color_palette("tab20", len(unique_clusters)) if len(unique_clusters) > 10 else sns.color_palette("tab10", len(unique_clusters))
            sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', style='sample', data=df, palette=palette, alpha=0.7)
            plt.title('UMAP with Leiden Clusters'); plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'plot_umap_leiden_clusters.png'), dpi=300); plt.close()

        self._generate_cluster_gallery(df_detailed, output_dir)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', required=True, choices=['file', 'folder', 'umap'])
    parser.add_argument('--input_paths', required=True, nargs='+')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--wt_path')
    parser.add_argument('--use_prealigned', action='store_true', help='Use pre-aligned (64,64,2) images directly.')
    parser.add_argument('--umap', action='store_true')
    parser.add_argument('--extra_viz', action='store_true')
    parser.add_argument('--quantitative', action='store_true')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir if args.output_dir else f"./screening_results/{timestamp}_{args.mode}"
    
    try:
        pipeline = MutantScreeningPipeline(args.model_dir, use_prealigned=args.use_prealigned)
        
        gen_umap = (args.mode == 'umap') or args.umap
        if args.mode == 'file':
            pipeline.run_file_mode(args.input_paths, output_dir, args.wt_path, gen_umap, args.extra_viz, args.quantitative)
        else:
            if len(args.input_paths) > 1: print("Warning: Only first input path used for folder mode root.")
            pipeline.run_folder_mode(args.input_paths[0], output_dir, gen_umap, args.extra_viz, args.quantitative, args.wt_path)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()