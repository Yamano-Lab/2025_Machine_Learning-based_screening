import numpy as np
import tifffile as tiff
import os
import argparse
import sys
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import tensorflow as tf

# GPU設定
os.environ['TF_DETERMINISTIC_OPS'] = '1'

class DualChannelAnomalyDetectionTraining:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
    def extract_quality_cells(self, image_path, stardist_model):
        try:
            image = tiff.imread(image_path)
            # 3D: (H, W, C) -> C=0: Red (Chl), C=1: Green (Pyr), C=2: Seg (sometimes)
            if image.ndim == 3 and image.shape[-1] >= 2:
                red_channel = image[..., 0]
                green_channel = image[..., 1]
                # If seg_channel is available use it, else use green
                seg_channel = image[..., 2] if image.shape[-1] >= 3 else image[..., 1]
            else:
                # Fallback for single channel (unlikely for dual channel task but safe handling)
                seg_channel = image
                green_channel = image
                red_channel = image
            
            normalized_seg = normalize(seg_channel)
            labels, _ = stardist_model.predict_instances(normalized_seg)
            
            height, width = labels.shape
            props = regionprops(labels)
            
            quality_cells = []
            cell_stats = []
            
            # Global normalization factors (per image)
            max_red = np.max(red_channel) if np.max(red_channel) > 0 else 1.0
            max_green = np.max(green_channel) if np.max(green_channel) > 0 else 1.0

            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)): continue
                if prop.area < 200 or prop.area > 8000: continue
                if prop.eccentricity > 0.95: continue
                
                # Crop both channels
                cell_red = red_channel[minr:maxr, minc:maxc]
                cell_green = green_channel[minr:maxr, minc:maxc]
                
                # Quality check based on Green Channel
                cell_mean = np.mean(cell_green)
                cell_std = np.std(cell_green)
                if cell_mean < 0.5 or cell_std < 0.1: continue
                
                # Resize and Normalize (0-1) independently
                # Note: Not using CLAHE here to preserve relative intensity differences if important,
                # or applying same CLAHE. 
                # The prompt asks for normalization. Let's stick to simple 0-1 scaling for now 
                # or CLAHE if consistent with previous. Previous used equalize_adapthist on Green.
                # Let's use equalize_adapthist for consistency and feature enhancement.
                
                # Red
                cell_red_float = cell_red / max_red
                cell_red_eq = exposure.equalize_adapthist(cell_red_float, clip_limit=0.02)
                cell_red_resized = resize(cell_red_eq, (64, 64), anti_aliasing=True)
                
                # Green
                cell_green_float = cell_green / max_green
                cell_green_eq = exposure.equalize_adapthist(cell_green_float, clip_limit=0.02)
                cell_green_resized = resize(cell_green_eq, (64, 64), anti_aliasing=True)
                
                # Stack to (64, 64, 2)
                cell_combined = np.stack([cell_red_resized, cell_green_resized], axis=-1)
                
                quality_cells.append(cell_combined)
                
                cell_stats.append({
                    'area': prop.area,
                    'file': os.path.basename(image_path)
                })
            return quality_cells, cell_stats
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], []
    
    def create_training_dataset(self):
        print(f"=== Creating Dual-Channel Dataset from: {self.input_dir} ===")
        stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        file_paths = sorted(glob(os.path.join(self.input_dir, '*.tif')) + glob(os.path.join(self.input_dir, '*.tiff')))
        
        all_cells = []
        all_stats = []
        
        for i, file_path in enumerate(file_paths):
            if i % 10 == 0: print(f"Processing {i+1}/{len(file_paths)}...")
            cells, stats = self.extract_quality_cells(file_path, stardist_model)
            all_cells.extend(cells)
            all_stats.extend(stats)
        
        print(f"\nTotal quality cells extracted: {len(all_cells)}")
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(self.output_dir, 'cell_statistics.csv'), index=False)
        return np.array(all_cells), stats_df
    
    def create_high_capacity_autoencoder(self, input_shape=(64, 64, 2)):
        """Dual-Channel Input/Output Autoencoder"""
        
        # --- Encoder (No Pooling: Strided Convolutions) ---
        input_img = Input(shape=input_shape, name='encoder_input')
        
        # Layer 1
        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Replace MaxPooling with Strided Conv
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 2
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Replace MaxPooling with Strided Conv
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 3
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Replace MaxPooling with Strided Conv
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        encoded = LeakyReLU(alpha=0.1, name='encoded_output')(x) # 8x8x64
        
        # --- Decoder ---
        encoded_input = Input(shape=(8, 8, 64), name='decoder_input')
        
        x = Conv2D(64, (3, 3), padding='same')(encoded_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        # Output has 2 channels (Red, Green)
        decoded_output = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='final_output')(x)
        
        # モデル構築
        encoder = Model(input_img, encoded, name='encoder')
        decoder = Model(encoded_input, decoded_output, name='decoder')
        autoencoder_output = decoder(encoder(input_img))
        autoencoder = Model(input_img, autoencoder_output, name='autoencoder')
        
        # --- Perceptual Loss (VGG19) ---
        # VGG19モデルの準備 (重み固定)
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        vgg.trainable = False
        # 特徴抽出用モデル (block3_conv3を使用)
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
        
        def perceptual_loss(y_true, y_pred):
            # 1. MAE Loss
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # 2. Perceptual Loss
            # Input is (Batch, 64, 64, 2). VGG needs 3 channels.
            # Pad with zeros for Blue channel.
            zeros = tf.zeros_like(y_true[..., 0:1])
            y_true_3ch = tf.concat([y_true, zeros], axis=-1)
            y_pred_3ch = tf.concat([y_pred, zeros], axis=-1)
            
            # 特徴量抽出
            feat_true = loss_model(y_true_3ch)
            feat_pred = loss_model(y_pred_3ch)
            
            # 特徴量間のMAE
            feat_loss = tf.reduce_mean(tf.abs(feat_true - feat_pred))
            
            # 組み合わせ (MAE + 0.1 * VGG_Loss)
            return mae + 0.1 * feat_loss

        # コンパイル
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss=perceptual_loss,
            metrics=['mse', 'mae']
        )
        
        return autoencoder, encoder, decoder
    
    def train_autoencoder(self, cell_images):
        print("=== Training Dual-Channel Autoencoder (Rotation Invariant) ===")
        
        # cell_images is already (N, 64, 64, 2)
        X = cell_images.astype('float32')
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 拡張あり (Rotation Invariance)
        datagen = ImageDataGenerator(
            rotation_range=180,      # Full rotation
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
        
        autoencoder, encoder, decoder = self.create_high_capacity_autoencoder()
        autoencoder.summary()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_autoencoder.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = autoencoder.fit(
            datagen.flow(X_train, X_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=300,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.plot_training_history(history)
        
        # 保存
        autoencoder.save(os.path.join(self.output_dir, 'final_autoencoder.keras'))
        encoder.save(os.path.join(self.output_dir, 'encoder.keras'))
        decoder.save(os.path.join(self.output_dir, 'decoder.keras'))
        
        return autoencoder, encoder, history
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss (Perceptual)')
        ax1.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()

    def create_anomaly_detector(self, encoder, cell_images):
        print("=== Creating Anomaly Detector (with Rotation Augmentation) ===")
        # Augment training data for SVM as well to ensure the SVM boundary 
        # covers rotated versions of normal cells.
        X = cell_images.astype('float32')
        
        # Simple augmentation for SVM: 0, 90, 180, 270 degrees
        X_0 = X
        X_90 = np.rot90(X, k=1, axes=(1, 2))
        X_180 = np.rot90(X, k=2, axes=(1, 2))
        X_270 = np.rot90(X, k=3, axes=(1, 2))
        X_aug = np.concatenate([X_0, X_90, X_180, X_270], axis=0)
        
        features = encoder.predict(X_aug, verbose=0)
        features_flat = features.reshape(len(features), -1)
        
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_flat)
        pca = PCA(n_components=min(100, features_scaled.shape[1], features_scaled.shape[0] - 1))
        features_reduced = pca.fit_transform(features_scaled)
        
        # Train One-Class SVM
        detectors = {'Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)}
        for name, detector in detectors.items():
            detector.fit(features_reduced)
        
        import pickle
        with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(self.output_dir, 'pca.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        for name, detector in detectors.items():
            with open(os.path.join(self.output_dir, f'detector_{name.lower()}.pkl'), 'wb') as f:
                pickle.dump(detector, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    trainer = DualChannelAnomalyDetectionTraining(args.input_dir, args.output_dir)
    cell_images, _ = trainer.create_training_dataset()
    if len(cell_images) < 100: return
    
    autoencoder, encoder, history = trainer.train_autoencoder(cell_images)
    trainer.create_anomaly_detector(encoder, cell_images)
    print(f"\n=== TRAINING COMPLETED ===")

if __name__ == "__main__":
    main()
