import numpy as np
import tifffile as tiff
import os
import argparse
import sys
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
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

class AlignedAnomalyDetectionTraining:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
    def load_aligned_dataset(self):
        print(f"=== Loading Aligned Dataset from: {self.input_dir} ===")
        file_paths = sorted(glob(os.path.join(self.input_dir, '*.tif')) + glob(os.path.join(self.input_dir, '*.tiff')))
        
        all_cells = []
        
        for i, file_path in enumerate(file_paths):
            if i % 100 == 0: print(f"Loading {i+1}/{len(file_paths)}...")
            try:
                # Expecting (64, 64, 2) float32 normalized images
                image = tiff.imread(file_path).astype('float32')
                if image.shape == (64, 64, 2):
                    all_cells.append(image)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"\nTotal aligned cells loaded: {len(all_cells)}")
        return np.array(all_cells)
    
    def create_high_capacity_autoencoder(self, input_shape=(64, 64, 2)):
        """Dual-Channel Input/Output Autoencoder"""
        
        # --- Encoder (High Resolution Preservation) ---
        input_img = Input(shape=input_shape, name='encoder_input')
        
        # Layer 1: 64x64 -> 32x32
        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 2: 32x32 -> 16x16
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 3: 16x16 -> 16x16 (No Downsampling to keep detail)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # Stride 1x1 here
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        encoded = LeakyReLU(alpha=0.1, name='encoded_output')(x) # 16x16x256
        
        # --- Decoder ---
        encoded_input = Input(shape=(16, 16, 256), name='decoder_input')
        
        # Layer 3 Reverse: 16x16 -> 16x16
        x = Conv2D(256, (3, 3), padding='same')(encoded_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # No UpSampling here
        
        # Layer 2 Reverse: 16x16 -> 32x32
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        # Layer 1 Reverse: 32x32 -> 64x64
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
        
        # Hybrid Loss (MSE + SSIM) for Structure Preservation
        def hybrid_loss(y_true, y_pred):
            # y_true/pred shape: (Batch, H, W, 2)
            # 0: Red (Chl), 1: Green (Pyr) 
            
            # 1. Weighted MSE (Color/Brightness)
            diff_sq = tf.square(y_true - y_pred)
            mse_red = tf.reduce_mean(diff_sq[..., 0])
            mse_green = tf.reduce_mean(diff_sq[..., 1])
            mse_term = mse_red + 20.0 * mse_green
            
            # 2. SSIM (Structure/Sharpness)
            # Calculate SSIM for each image in batch and average
            ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
            ssim_loss = 1.0 - tf.reduce_mean(ssim_val)
            
            # Combine: MSE + 2.0 * SSIM
            return mse_term + 2.0 * ssim_loss

        # コンパイル
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss=hybrid_loss,
            metrics=['mse', 'mae']
        )
        
        return autoencoder, encoder, decoder
    
    def visualize_2ch(self, image_data):
        """
        Visualize (64, 64, 2) as an RGB image.
        R=Red Ch, G=Green Ch, B=0
        """
        h, w, c = image_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if c >= 1: rgb[..., 0] = image_data[..., 0] # Red
        if c >= 2: rgb[..., 1] = image_data[..., 1] # Green
        return np.clip(rgb, 0, 1)

    def save_reconstruction_examples(self, autoencoder, X_val, filename='reconstruction_samples.png'):
        print(f"Generating reconstruction examples to {filename}...")
        n = 10
        # Select random samples
        indices = np.random.choice(len(X_val), n, replace=False) if len(X_val) > n else np.arange(len(X_val))
        samples = X_val[indices]
        reconstructed = autoencoder.predict(samples, verbose=0)
        
        plt.figure(figsize=(20, 4))
        for i in range(len(samples)):
            # Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.visualize_2ch(samples[i]))
            plt.title("Input (Aligned)")
            plt.axis("off")
            
            # Reconstructed
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(self.visualize_2ch(reconstructed[i]))
            plt.title("Reconstructed")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    def train_autoencoder(self, cell_images):
        print("=== Training Aligned Autoencoder (Structure Preserving) ===")
        
        # cell_images is already (N, 64, 64, 2)
        X = cell_images.astype('float32')
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 厳格なAugmentation設定 (回転禁止、上下反転禁止)
        datagen = ImageDataGenerator(
            rotation_range=0,        # No rotation
            width_shift_range=0.05,  # Minimal shift
            height_shift_range=0.05, # Minimal shift
            horizontal_flip=True,    # L/R flip is OK
            vertical_flip=False,     # No U/D flip (pyrenoid at bottom)
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
        
        # Save reconstruction examples
        self.save_reconstruction_examples(autoencoder, X_val)
        
        # 保存
        autoencoder.save(os.path.join(self.output_dir, 'final_autoencoder.keras'))
        encoder.save(os.path.join(self.output_dir, 'encoder.keras'))
        decoder.save(os.path.join(self.output_dir, 'decoder.keras'))
        
        return autoencoder, encoder, history
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss (Hybrid: MSE + SSIM)')
        ax1.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()

    def plot_anomaly_score_distribution(self, scores, filename='anomaly_score_dist.png'):
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, color='blue', alpha=0.7, log=True)
        plt.title('Anomaly Score (MSE) Distribution (Log Scale)')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count (Log)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def create_anomaly_detector(self, autoencoder, encoder, cell_images):
        print("=== Creating Anomaly Detector ===")
        
        # Calculate MSE for anomaly score distribution
        print("Calculating MSE distribution for training data...")
        X = cell_images.astype('float32')
        reconstructions = autoencoder.predict(X, verbose=0)
        mse_scores = np.mean(np.square(X - reconstructions), axis=(1, 2, 3))
        self.plot_anomaly_score_distribution(mse_scores)
        
        # SVM Training (No Augmentation needed for aligned data)
        features = encoder.predict(X, verbose=0)
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
    
    trainer = AlignedAnomalyDetectionTraining(args.input_dir, args.output_dir)
    cell_images = trainer.load_aligned_dataset()
    if len(cell_images) < 100: return
    
    autoencoder, encoder, history = trainer.train_autoencoder(cell_images)
    trainer.create_anomaly_detector(autoencoder, encoder, cell_images)
    print(f"\n=== TRAINING COMPLETED ===")

if __name__ == "__main__":
    main()
