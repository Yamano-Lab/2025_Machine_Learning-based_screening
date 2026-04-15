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

# --- Extended Analysis Imports ---
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance, mannwhitneyu
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
    変異株スクリーニングの統合パイプライン (Aligned Model 対応版) v4
    - 要件定義書対応版: 
        1. 多次元表現型スコアリング (PH-001)
        2. 統計的比較とHit株同定 (ST-001, ST-002)
        3. 面積フィルタリング緩和・スコア化 (v3継承)
        4. ロバストなアライメント (v3継承)
        5. Leidenクラスタリング (v3継承)
    """

    def __init__(self, model_dir, use_prealigned=False):
        self.model_dir = model_dir
        self.use_prealigned = use_prealigned
        
        # --- Visual Style Setup ---
        self.okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        sns.set_palette(self.okabe_ito)
        sns.set_style("ticks")
        plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.autolayout': False})
        
        self.load_trained_models()

    def load_trained_models(self):
        """訓練済みAIモデルと関連ファイルを読み込む"""
        print("Loading trained models...")
        try:
            self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'), compile=False)
            
            # Extract Decoder
            maxpool_idx = -1
            for i, layer in enumerate(self.autoencoder.layers):
                if isinstance(layer, MaxPooling2D): maxpool_idx = i
            if maxpool_idx != -1:
                decoder_layers = self.autoencoder.layers[maxpool_idx+1:]
                decoder_input = Input(shape=(8, 8, 32))
                x = decoder_input
                for layer in decoder_layers: x = layer(x)
                self.decoder = Model(decoder_input, x)

            self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'), compile=False)
            
            # Load Scaler/PCA
            for v in ['v2', '']:
                p = os.path.join(self.model_dir, f'scaler{("_"+v) if v else ""}.pkl')
                if os.path.exists(p): self.scaler = pickle.load(open(p, 'rb')); break
            for v in ['v2', '']:
                p = os.path.join(self.model_dir, f'pca{("_"+v) if v else ""}.pkl')
                if os.path.exists(p): self.pca = pickle.load(open(p, 'rb')); break

            # Load Detector
            for n in ['detector_svm_v2.pkl', 'detector_conservative.pkl']:
                p = os.path.join(self.model_dir, n)
                if os.path.exists(p): 
                    self.detector_conservative = pickle.load(open(p, 'rb'))
                    break
            else:
                self.detector_conservative = None
                
            self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}"); raise

    def extract_quality_cells(self, image_path, enhance_contrast=False):
        try:
            if self.use_prealigned:
                    try:
                        image = tiff.imread(image_path)
                        if image.shape == (64, 64, 2):
                            # Prealigned feature extraction
                            red = image[..., 0]
                            green = image[..., 1]
                            
                            # Simple mask for features (Red channel > 0.1)
                            mask = red > 0.1
                            area = np.sum(mask)
                            
                            if area < 10: return [] # Too small
                            
                            intensity = np.mean(green[mask]) if area > 0 else 0
                            
                            # Localization (Green > 0.2)
                            green_mask = green > 0.2
                            loc_score = 32.0
                            if np.any(green_mask):
                                try:
                                    gy, gx = regionprops(green_mask.astype(int))[0].centroid
                                    loc_score = math.sqrt((gx - 32)**2 + (gy - 48)**2)
                                except: pass
                                
                            # RG Ratio
                            sum_red = np.sum(red)
                            sum_green = np.sum(green)
                            rg_ratio = sum_red / (sum_green + 1e-6)
                            
                            # Compactness
                            compactness = 0
                            try:
                                props_aligned = regionprops(mask.astype(int))
                                if props_aligned:
                                    p = props_aligned[0].perimeter
                                    a = props_aligned[0].area
                                    if p > 0: compactness = (4 * math.pi * a) / (p**2)
                            except: pass
    
                            features = {
                                'pyrenoid_size': float(area),
                                'pyrenoid_intensity': float(intensity),
                                'pyrenoid_compactness': float(compactness),
                                'localization_score': float(loc_score),
                                'red_green_ratio': float(rg_ratio),
                                'align_method': 'prealigned',
                                'size_z_score': 0.0
                            }
                            return [(image, image, features)]
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
            else: return []

            normalized_seg = normalize(seg_channel)
            labels, _ = self.stardist_model.predict_instances(normalized_seg)
            props = regionprops(labels)
            
            quality_cells = []
            p99_red = np.percentile(red_channel, 99.9)
            p99_green = np.percentile(green_channel, 99.9)
            
            H, W = red_channel.shape
            
            for prop in props:
                # 1. Filtering
                minr, minc, maxr, maxc = prop.bbox
                if minr < 5 or minc < 5 or maxr > H - 5 or maxc > W - 5: continue
                if prop.area < 50: continue # Hard garbage filter
                if prop.eccentricity > 0.98: continue

                # 2. Crop
                cy, cx = map(int, prop.centroid)
                size = max(int(max(maxr-minr, maxc-minc) * 1.5), 64)
                half = size // 2
                
                # ... (Padding/Crop logic omitted for brevity, assuming valid crop) ...
                # Re-implementing simplified valid crop for robustness in this snippet
                r_start, c_start = cy - half, cx - half
                r_end, c_end = r_start + size, c_start + size
                
                # Check bounds
                if r_start < 0 or c_start < 0 or r_end > H or c_end > W: continue # Skip edge cases for simplicity in v4

                crop_red = red_channel[r_start:r_end, c_start:c_end]
                crop_green = green_channel[r_start:r_end, c_start:c_end]
                mask_slice = (labels[r_start:r_end, c_start:c_end] == prop.label)
                
                crop_red = crop_red * mask_slice
                crop_green = crop_green * mask_slice

                # 3. Alignment
                align_method = 'pyrenoid'
                angle = 0
                try:
                    m = regionprops(mask_slice.astype(int), intensity_image=crop_red)[0]
                    cy_crop, cx_crop = m.centroid
                    
                    max_green = np.max(crop_green)
                    if max_green >= p99_green * 0.1:
                        mg = regionprops(mask_slice.astype(int), intensity_image=crop_green)[0]
                        py, px = mg.weighted_centroid
                        dy, dx = py - cy_crop, px - cx_crop
                        if dy**2 + dx**2 >= 2.0:
                            angle = math.degrees(math.atan2(dy, dx)) - 90
                    else:
                        align_method = 'axis'
                        angle = -math.degrees(m.orientation)
                except:
                    continue

                final_crop_size = 64
                aligned_red = resize(rotate(crop_red, angle, resize=False), (final_crop_size, final_crop_size))
                aligned_green = resize(rotate(crop_green, angle, resize=False), (final_crop_size, final_crop_size))
                
                # Normalize
                norm_red = np.clip(aligned_red / p99_red, 0, 1)
                norm_green = np.clip(aligned_green / p99_green, 0, 1)
                aligned_cell = np.stack([norm_red, norm_green], axis=-1).astype(np.float32)

                # --- PH-001: Phenotype Quantification ---
                # 1. Area
                area = prop.area
                # 2. Intensity
                intensity = prop.mean_intensity if hasattr(prop, 'mean_intensity') else np.mean(crop_green[mask_slice])
                # 3. Compactness (4*pi*A / P^2)
                compactness = (4 * math.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
                # 4. Localization Score (Distance from bottom-center (32, 48))
                # Note: Normalized image coordinates are used.
                # Find centroid of green in aligned image
                try:
                    green_mask = norm_green > 0.2 # Threshold for pyrenoid
                    if np.any(green_mask):
                        gy, gx = regionprops(green_mask.astype(int))[0].centroid
                        # Target: x=32, y=48 (64x64 image)
                        loc_score = math.sqrt((gx - 32)**2 + (gy - 48)**2)
                    else:
                        loc_score = 32.0 # Default penalty
                except:
                    loc_score = 32.0
                
                # 5. Red/Green Ratio
                sum_red = np.sum(crop_red)
                sum_green = np.sum(crop_green)
                rg_ratio = sum_red / (sum_green + 1e-6)

                features = {
                    'pyrenoid_size': area,
                    'pyrenoid_intensity': intensity,
                    'pyrenoid_compactness': compactness,
                    'localization_score': loc_score,
                    'red_green_ratio': rg_ratio,
                    'align_method': align_method
                }
                
                quality_cells.append((aligned_cell, aligned_cell, features))
                
            return quality_cells
        except Exception as e:
            print(f"Error: {e}"); return []

    def analyze_size_distribution(self, all_data, output_dir):
        """サイズ分布解析とZ-score計算 (v3継承)"""
        print("\n--- Analyzing Size Distribution ---")
        wt_data = [d['features']['pyrenoid_size'] for d in all_data if 'WT' in d['sample'].upper() or d['sample']=='WT']
        
        if not wt_data:
            print("  [Warning] WT not found. Skipping size Z-score.")
            return all_data
            
        wt_mean = np.mean(wt_data)
        wt_std = np.std(wt_data)
        print(f"  WT Area: Mean={wt_mean:.1f}, Std={wt_std:.1f}")
        
        kept_data = []
        for d in all_data:
            size = d['features']['pyrenoid_size']
            d['features']['size_z_score'] = (size - wt_mean) / (wt_std + 1e-6)
            # Soft filter: remove extreme garbage
            if 50 <= size <= 5000: kept_data.append(d)
        
        print(f"  Cells Kept: {len(kept_data)} / {len(all_data)}")
        
        # Save Statistics (DS-002)
        stats = {
            'total_detected': len(all_data),
            'total_kept': len(kept_data),
            'garbage_removed': len(all_data) - len(kept_data),
            'acceptance_rate': len(kept_data) / len(all_data) if all_data else 0
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'filtering_statistics.csv'), index=False)
        
        # Plot
        df_plot = pd.DataFrame([{'sample': d['sample'], 'size': d['features']['pyrenoid_size'], 'type': 'WT' if 'WT' in d['sample'] else 'Mut'} for d in kept_data])
        plt.figure(figsize=(10,6))
        sns.kdeplot(data=df_plot[df_plot['type']=='WT'], x='size', fill=True, color='gray', label='WT')
        plt.title('Size Distribution'); plt.savefig(os.path.join(output_dir, 'size_dist_v4.png')); plt.close()
        return kept_data

    def perform_statistical_analysis(self, df_detailed, output_dir):
        """
        ST-001, ST-002: 統計的比較とHit株同定
        """
        print("\n--- Performing Statistical Analysis (ST-001) ---")
        wt_name = next((s for s in df_detailed['sample_name'].unique() if s.upper()=='WT'), None)
        if not wt_name: print("  [Error] WT not found for stats."); return

        wt_df = df_detailed[df_detailed['sample_name'] == wt_name]
        mutants = [s for s in df_detailed['sample_name'].unique() if s != wt_name]
        
        metrics = ['anomaly_score', 'mse', 'pyrenoid_size', 'pyrenoid_intensity', 'pyrenoid_compactness', 'localization_score', 'red_green_ratio']
        results = []

        for mut in mutants:
            mut_df = df_detailed[df_detailed['sample_name'] == mut]
            for metric in metrics:
                if metric not in df_detailed.columns: continue
                
                wt_vals = wt_df[metric].dropna()
                mut_vals = mut_df[metric].dropna()
                
                if len(wt_vals) < 3 or len(mut_vals) < 3: continue
                
                # Mann-Whitney U test
                stat, p_val = mannwhitneyu(wt_vals, mut_vals, alternative='two-sided')
                
                # Cohen's d
                n1, n2 = len(wt_vals), len(mut_vals)
                var1, var2 = np.var(wt_vals, ddof=1), np.var(mut_vals, ddof=1)
                pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                cohens_d = (np.mean(mut_vals) - np.mean(wt_vals)) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'strain_id': mut,
                    'phenotype': metric,
                    'p_value': p_val,
                    'effect_size': cohens_d,
                    'wt_mean': np.mean(wt_vals),
                    'mutant_mean': np.mean(mut_vals)
                })

        if not results: return
        
        stats_df = pd.DataFrame(results)
        
        # FDR Correction
        if STATSMODELS_AVAILABLE:
            _, stats_df['p_adjusted'], _, _ = multipletests(stats_df['p_value'], method='fdr_bh')
        else:
            stats_df['p_adjusted'] = stats_df['p_value'] * len(stats_df) # Bonferroni fallback
            stats_df['p_adjusted'] = stats_df['p_adjusted'].clip(upper=1.0)

        stats_df['is_significant'] = stats_df['p_adjusted'] < 0.05
        stats_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
        
        # Hit Identification (ST-002)
        hits = stats_df[(stats_df['p_adjusted'] < 0.05) & (stats_df['effect_size'].abs() > 0.8)]
        hits.to_csv(os.path.join(output_dir, 'hit_strains.csv'), index=False)
        print(f"  Identified {len(hits['strain_id'].unique())} potential hit strains.")

    def run_folder_mode(self, root_path, output_dir, generate_umap, run_extra_viz, run_quantitative, wt_path=None):
        print(f"\n=== Running in FOLDER mode (v4: Requirements Compliant) ===")
        folders_dict = self._get_folders_from_path(root_path)
        if not folders_dict: print("No subfolders found."); return
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load Data
        all_data = [] 
        for name, folder_path in folders_dict.items():
            print(f"  Loading: {name}...", end='\r')
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')))
            for f_path in tif_files:
                cells = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw, pre, feats) in enumerate(cells):
                    all_data.append({'sample': name, 'path': f_path, 'id': i, 'raw': raw, 'pre': pre, 'features': feats})
        
        # External WT
        if wt_path and not any(d['sample']=='WT' for d in all_data):
            print("\n  Loading External WT...")
            wt_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff')))
            for f_path in wt_files:
                cells = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw, pre, feats) in enumerate(cells):
                    all_data.append({'sample': 'WT', 'path': f_path, 'id': i, 'raw': raw, 'pre': pre, 'features': feats})

        # 2. Analyze Size
        filtered_data = self.analyze_size_distribution(all_data, output_dir)
        
        # 3. Compute Scores
        results, analysis_data = [], {'features': [], 'sample': [], 'is_anomaly': [], 'mse': []}
        
        unique_samples = sorted(list(set(d['sample'] for d in filtered_data)))
        for name in unique_samples:
            sample_data = [d for d in filtered_data if d['sample'] == name]
            if not sample_data: continue
            
            # Compute Autoencoder Scores
            images = [d['pre'] for d in sample_data]
            # (Assuming compute_anomaly_scores exists and is similar to v3, re-implementing inline/simplified for context)
            X = np.array(images).astype('float32')
            
            # Batch predict
            enc_feats = self.encoder.predict(X, verbose=0)
            enc_flat = enc_feats.reshape(len(X), -1)
            reconst = self.autoencoder.predict(X, verbose=0)
            mse = np.mean(np.square(X - reconst), axis=(1,2,3))
            
            # PCA/Scaler
            scaled = self.scaler.transform(enc_flat)
            pca_feats = self.pca.transform(scaled)
            
            # Anomaly Score
            if self.detector_conservative:
                scores = -self.detector_conservative.decision_function(pca_feats)
            else:
                scores = mse

            # Store Results
            for i, d in enumerate(sample_data):
                row = {
                    'sample_name': name, 'file_path': d['path'], 'cell_id': d['id'],
                    'anomaly_score': scores[i], 'mse': mse[i],
                    # Add new features
                    'pyrenoid_size': d['features']['pyrenoid_size'],
                    'pyrenoid_intensity': d['features']['pyrenoid_intensity'],
                    'pyrenoid_compactness': d['features']['pyrenoid_compactness'],
                    'localization_score': d['features']['localization_score'],
                    'red_green_ratio': d['features']['red_green_ratio'],
                    'size_z_score': d['features']['size_z_score'],
                    'align_method': d['features']['align_method']
                }
                results.append(row)
            
            if generate_umap or run_quantitative:
                analysis_data['features'].append(pca_feats)
                analysis_data['sample'].extend([name]*len(X))
                analysis_data['mse'].extend(mse)

        df_detailed = pd.DataFrame(results)
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_v4.csv'), index=False)
        
        # 4. Statistical Analysis
        if run_quantitative and len(results) > 0:
            self.perform_statistical_analysis(df_detailed, output_dir)
            
            # Clustering (Leiden)
            if analysis_data['features']:
                all_feats = np.concatenate(analysis_data['features'], axis=0)
                # Call clustering (reuse v3 logic logic here essentially)
                self.perform_clustering_analysis(all_feats, pd.DataFrame({'sample': analysis_data['sample']}), output_dir, None, df_detailed)

        # 5. UMAP
        if generate_umap and analysis_data['features']:
            all_feats = np.concatenate(analysis_data['features'], axis=0)
            self.create_umap_visualization(all_feats, pd.DataFrame({'sample': analysis_data['sample'], 'mse': analysis_data['mse']}), output_dir, None)

        print(f"Processing complete. Results: {output_dir}")

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

    def run_file_mode(self, input_paths, output_dir, wt_path=None, generate_umap=False, run_extra_viz=False, run_quantitative=False):
        print(f"\n=== Running in FILE mode (v4) ===")
        files_dict = self._get_files_from_paths(input_paths)
        if not files_dict: print("No TIF files found."); return
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = [] 
        for name, path in files_dict.items():
            print(f"  Processing: {name}...", end='\r')
            cells = self.extract_quality_cells(path, enhance_contrast=True)
            for i, (raw, pre, feats) in enumerate(cells):
                all_data.append({'sample': name, 'path': path, 'id': i, 'raw': raw, 'pre': pre, 'features': feats})
        
        if wt_path and not any(d['sample']=='WT' for d in all_data):
            wt_files = sorted(glob(os.path.join(wt_path, '*.tif')) + glob(os.path.join(wt_path, '*.tiff'))) if os.path.isdir(wt_path) else [wt_path]
            for f_path in wt_files:
                cells = self.extract_quality_cells(f_path, enhance_contrast=True)
                for i, (raw, pre, feats) in enumerate(cells):
                    all_data.append({'sample': 'WT', 'path': f_path, 'id': i, 'raw': raw, 'pre': pre, 'features': feats})

        filtered_data = self.analyze_size_distribution(all_data, output_dir)
        
        results, analysis_data = [], {'features': [], 'sample': [], 'mse': []}
        
        unique_samples = sorted(list(set(d['sample'] for d in filtered_data)))
        for name in unique_samples:
            sample_data = [d for d in filtered_data if d['sample'] == name]
            if not sample_data: continue
            
            images = [d['pre'] for d in sample_data]
            X = np.array(images).astype('float32')
            
            enc_feats = self.encoder.predict(X, verbose=0)
            enc_flat = enc_feats.reshape(len(X), -1)
            reconst = self.autoencoder.predict(X, verbose=0)
            mse = np.mean(np.square(X - reconst), axis=(1,2,3))
            
            scaled = self.scaler.transform(enc_flat)
            pca_feats = self.pca.transform(scaled)
            
            scores = -self.detector_conservative.decision_function(pca_feats) if self.detector_conservative else mse

            for i, d in enumerate(sample_data):
                row = {
                    'sample_name': name, 'file_path': d['path'], 'cell_id': d['id'],
                    'anomaly_score': scores[i], 'mse': mse[i],
                    'pyrenoid_size': d['features']['pyrenoid_size'],
                    'pyrenoid_intensity': d['features']['pyrenoid_intensity'],
                    'pyrenoid_compactness': d['features']['pyrenoid_compactness'],
                    'localization_score': d['features']['localization_score'],
                    'red_green_ratio': d['features']['red_green_ratio'],
                    'size_z_score': d['features']['size_z_score'],
                    'align_method': d['features']['align_method']
                }
                results.append(row)
            
            if generate_umap or run_quantitative:
                analysis_data['features'].append(pca_feats)
                analysis_data['sample'].extend([name]*len(X))
                analysis_data['mse'].extend(mse)

        df_detailed = pd.DataFrame(results)
        df_detailed.to_csv(os.path.join(output_dir, 'detailed_results_v4.csv'), index=False)
        
        if run_quantitative and len(results) > 0:
            self.perform_statistical_analysis(df_detailed, output_dir)
            if analysis_data['features']:
                all_feats = np.concatenate(analysis_data['features'], axis=0)
                self.perform_clustering_analysis(all_feats, pd.DataFrame({'sample': analysis_data['sample']}), output_dir, None, df_detailed)

        if generate_umap and analysis_data['features']:
            all_feats = np.concatenate(analysis_data['features'], axis=0)
            self.create_umap_visualization(all_feats, pd.DataFrame({'sample': analysis_data['sample'], 'mse': analysis_data['mse']}), output_dir, None)

        print(f"Processing complete. Results: {output_dir}")

    # --- Helpers (Stubbed/Simplified from v3) ---
    def _get_folders_from_path(self, root_path):
        if not os.path.isdir(root_path): return {}
        target_folders = {}
        for root, dirs, files in os.walk(root_path):
            if any(f.endswith(('.tif', '.tiff')) for f in files):
                name = os.path.basename(root)
                if name in target_folders: name = f"{name}_{os.path.basename(os.path.dirname(root))}"
                target_folders[name] = root
        return target_folders

    def perform_clustering_analysis(self, features, df, output_dir, color_map, df_detailed):
        print("  Performing Clustering...")
        if SCANPY_AVAILABLE:
            adata = anndata.AnnData(X=features)
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, resolution=0.5)
            df_detailed['cluster'] = adata.obs['leiden'].values
        else:
            from sklearn.cluster import KMeans
            df_detailed['cluster'] = KMeans(n_clusters=10, random_state=42).fit_predict(features)
        
        # Save composition
        comp = df_detailed.groupby(['sample_name', 'cluster']).size().unstack(fill_value=0)
        comp_perc = comp.div(comp.sum(axis=1), axis=0) * 100
        comp_perc.to_csv(os.path.join(output_dir, 'cluster_composition.csv'))
        
        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(comp_perc, cmap='viridis'); plt.savefig(os.path.join(output_dir, 'cluster_heatmap.png')); plt.close()

    def create_umap_visualization(self, features, df, output_dir, color_map):
        print("  Generating UMAP...")
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        plt.figure(figsize=(10,8))
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=df['sample'], s=10, alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'umap_v4.png')); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Integrated Mutant Screening v4")
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
    output_dir = args.output_dir if args.output_dir else f"./screening_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_v4"
    
    pipeline = MutantScreeningPipeline(args.model_dir, use_prealigned=args.use_prealigned)
    
    gen_umap = (args.mode == 'umap') or args.umap
    
    if args.mode == 'file':
        pipeline.run_file_mode(args.input_paths, output_dir, args.wt_path, gen_umap, args.extra_viz, args.quantitative)
    elif args.mode in ['folder', 'umap']:
        if len(args.input_paths) > 1: print("Warning: Using first input path only.")
        pipeline.run_folder_mode(args.input_paths[0], output_dir, gen_umap, args.extra_viz, args.quantitative, args.wt_path)

if __name__ == "__main__":
    main()
