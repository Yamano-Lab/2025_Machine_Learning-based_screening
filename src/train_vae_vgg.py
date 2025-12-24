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
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

# GPU設定
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- Custom Layer for Reparameterization Trick ---
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- VAE Model with Custom Training Step ---
class VAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, gamma=0.00002, recon_weight=100.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.gamma = gamma
        self.recon_weight = recon_weight
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
        
        # Setup VGG for Perceptual Loss
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        vgg.trainable = False
        # Extract features from block3_conv3
        self.vgg_feature_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        self.vgg_feature_model.trainable = False

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.vgg_loss_tracker,
        ]

    def train_step(self, data):
        # Unpack data (ImageDataGenerator yields (x, x))
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # --- Loss Calculation ---
            
            # 1. Reconstruction Loss (MAE) - Raw calculation
            recon_loss_raw = tf.reduce_mean(tf.abs(data - reconstruction))
            
            # 2. KL Divergence Loss - Raw calculation
            kl_loss_raw = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss_raw = tf.reduce_mean(tf.reduce_sum(kl_loss_raw, axis=1))
            
            # 3. Perceptual Loss (VGG) - Raw calculation
            # Preprocess for VGG: Scale to 0-255, 3 channels, specific preprocessing
            data_scaled = data * 255.0
            recon_scaled = reconstruction * 255.0
            
            data_rgb = tf.image.grayscale_to_rgb(data_scaled)
            recon_rgb = tf.image.grayscale_to_rgb(recon_scaled)
            
            data_preprocessed = preprocess_input(data_rgb)
            recon_preprocessed = preprocess_input(recon_rgb)
            
            feat_true = self.vgg_feature_model(data_preprocessed)
            feat_pred = self.vgg_feature_model(recon_preprocessed)
            
            vgg_loss_raw = tf.reduce_mean(tf.square(feat_true - feat_pred))
            
            # Total Loss (Weighted Sum)
            total_loss = (self.recon_weight * recon_loss_raw) + \
                         (self.beta * kl_loss_raw) + \
                         (self.gamma * vgg_loss_raw)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Track raw or weighted? Usually track weighted components to see contribution, or raw to see physical meaning.
        # Let's track the weighted contributions as per Keras standard logic usually, but here raw is more informative if we know weights.
        # However, to be consistent with 'total_loss', let's track the terms that sum to it.
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss_raw) # Tracking raw MAE for interpretability
        self.kl_loss_tracker.update_state(kl_loss_raw) # Tracking raw KL
        self.vgg_loss_tracker.update_state(vgg_loss_raw) # Tracking raw VGG MSE
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "vgg_loss": self.vgg_loss_tracker.result(),
        }
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Raw calculations
        recon_loss_raw = tf.reduce_mean(tf.abs(data - reconstruction))
        
        kl_loss_raw = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss_raw = tf.reduce_mean(tf.reduce_sum(kl_loss_raw, axis=1))
        
        data_scaled = data * 255.0
        recon_scaled = reconstruction * 255.0
        data_rgb = tf.image.grayscale_to_rgb(data_scaled)
        recon_rgb = tf.image.grayscale_to_rgb(recon_scaled)
        data_preprocessed = preprocess_input(data_rgb)
        recon_preprocessed = preprocess_input(recon_rgb)
        
        feat_true = self.vgg_feature_model(data_preprocessed)
        feat_pred = self.vgg_feature_model(recon_preprocessed)
        
        vgg_loss_raw = tf.reduce_mean(tf.square(feat_true - feat_pred))

        # Total Loss
        total_loss = (self.recon_weight * recon_loss_raw) + \
                     (self.beta * kl_loss_raw) + \
                     (self.gamma * vgg_loss_raw)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss_raw)
        self.kl_loss_tracker.update_state(kl_loss_raw)
        self.vgg_loss_tracker.update_state(vgg_loss_raw)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "vgg_loss": self.vgg_loss_tracker.result(),
        }

class VAETrainer:
    def __init__(self, input_dir, output_dir, latent_dim=128, beta=1.0, gamma=0.00002, recon_weight=100.0):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.recon_weight = recon_weight
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
    def extract_quality_cells(self, image_path, stardist_model):
        """Extracts single cells from large microscopy images using StarDist."""
        try:
            image = tiff.imread(image_path)
            # Handle dimensions
            if image.ndim == 3 and image.shape[-1] >= 3:
                seg_channel = image[..., 2] # Assuming DAPI/Nuclear channel
                green_channel = image[..., 1] # Target channel
            else:
                seg_channel = image
                green_channel = image
            
            normalized_seg = normalize(seg_channel)
            labels, _ = stardist_model.predict_instances(normalized_seg)
            
            height, width = labels.shape
            props = regionprops(labels)
            
            quality_cells = []
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                # Boundary check
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)): continue
                # Size check
                if prop.area < 200 or prop.area > 8000: continue
                # Shape check
                if prop.eccentricity > 0.95: continue
                
                cell_image = green_channel[minr:maxr, minc:maxc]
                
                # Intensity check
                if np.mean(cell_image) < 0.5 or np.std(cell_image) < 0.1: continue
                
                # Preprocessing
                cell_image_eq = exposure.equalize_adapthist(cell_image, clip_limit=0.02)
                cell_image_resized = resize(cell_image_eq, (64, 64), anti_aliasing=True)
                quality_cells.append(cell_image_resized)
                
            return quality_cells
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def create_dataset(self):
        print(f"=== Creating Dataset from: {self.input_dir} ===")
        # Explicitly ignore warning
        import logging
        tf.get_logger().setLevel(logging.ERROR)
        
        stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        file_paths = sorted(glob(os.path.join(self.input_dir, '*.tif')) + glob(os.path.join(self.input_dir, '*.tiff')))
        
        if not file_paths:
            print("No TIFF files found in input directory.")
            return np.array([])

        all_cells = []
        for i, file_path in enumerate(file_paths):
            if i % 10 == 0: print(f"Processing {i+1}/{len(file_paths)}...")
            cells = self.extract_quality_cells(file_path, stardist_model)
            all_cells.extend(cells)
        
        print(f"\nTotal cells extracted: {len(all_cells)}")
        return np.array(all_cells)

    def build_models(self, input_shape=(64, 64, 1)):
        # --- Encoder ---
        encoder_inputs = Input(shape=input_shape, name="encoder_input")
        
        x = Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
        x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = Conv2D(128, 3, strides=2, padding="same", activation="relu")(x) # 8x8x128
        
        shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
        
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
        # --- Decoder ---
        latent_inputs = Input(shape=(self.latent_dim,), name="z_sampling")
        x = Dense(np.prod(shape_before_flattening), activation="relu")(latent_inputs)
        x = Reshape(shape_before_flattening)(x)
        
        x = Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        decoder_outputs = Conv2DTranspose(1, 3, strides=1, padding="same", activation="sigmoid")(x)
        
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")
        
        return encoder, decoder

    def train(self, epochs=100, batch_size=32):
        cells = self.create_dataset()
        if len(cells) == 0:
            print("No cells found or generated. Exiting.")
            return

        # Prepare Data
        X = np.expand_dims(cells, axis=-1).astype('float32')
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True
        )
        
        # Build VAE
        encoder, decoder = self.build_models()
        vae = VAE(encoder, decoder, beta=self.beta, gamma=self.gamma, recon_weight=self.recon_weight)
        
        vae.compile(optimizer=Adam(learning_rate=0.0005))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        print(f"=== Starting VAE Training (Beta={self.beta}, Gamma={self.gamma}, Recon_Weight={self.recon_weight}) ===")

        # Create generator
        train_generator = datagen.flow(X_train, X_train, batch_size=batch_size)
        
        # (Optional but recommended safety) Verify generator and reset
        try:
            sample_batch = next(train_generator)
            print(f"Generator check: Batch shape {sample_batch[0].shape}")
        except Exception as e:
            print(f"Generator check failed: {e}")

        # RESET GENERATOR to avoid "ran out of data" error
        train_generator = datagen.flow(X_train, X_train, batch_size=batch_size)

        history = vae.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, X_val),
            callbacks=callbacks
        )
        
        self.save_results(vae, encoder, decoder, history, X_val)

    def save_results(self, vae, encoder, decoder, history, X_val):
        print("=== Saving Results ===")
        # Save Models
        encoder.save(os.path.join(self.output_dir, 'vae_encoder.keras'))
        decoder.save(os.path.join(self.output_dir, 'vae_decoder.keras'))
        
        try:
            vae.save_weights(os.path.join(self.output_dir, 'vae_weights.weights.h5'))
        except Exception as e:
            print(f"Could not save VAE weights: {e}")

        # Save History
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(os.path.join(self.output_dir, 'training_history.csv'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(hist_df['loss'], label='Total Loss')
        plt.plot(hist_df['val_loss'], label='Val Loss')
        plt.plot(hist_df['recon_loss'], label='Recon Loss', linestyle='--')
        plt.plot(hist_df['vgg_loss'], label='VGG Loss', linestyle='--')
        plt.title('VAE Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
        plt.close()
        
        # Save Reconstructions
        n_samples = 10
        samples = X_val[:n_samples]
        # VAE output is reconstruction
        # We need to manually pass through encoder/decoder or call vae.predict (might need dummy labels)
        _, _, z_sample = encoder.predict(samples)
        reconstructed = decoder.predict(z_sample)
        
        plt.figure(figsize=(20, 4))
        for i in range(n_samples):
            # Original
            ax = plt.subplot(2, n_samples, i + 1)
            plt.imshow(samples[i].reshape(64, 64), cmap='gray')
            plt.title("Original")
            plt.axis("off")
            
            # Reconstructed
            ax = plt.subplot(2, n_samples, i + 1 + n_samples)
            plt.imshow(reconstructed[i].reshape(64, 64), cmap='gray')
            plt.title("Reconstructed")
            plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reconstructions.png'))
        plt.close()
        print(f"Results saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train VAE with VGG Perceptual Loss")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input TIFF files")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save outputs")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--beta', type=float, default=1.0, help="Weight for KL Divergence")
    parser.add_argument('--gamma', type=float, default=0.00002, help="Weight for VGG Perceptual Loss")
    parser.add_argument('--recon_weight', type=float, default=100.0, help="Weight for Reconstruction Loss (MAE)")
    
    args = parser.parse_args()
    
    trainer = VAETrainer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        beta=args.beta,
        gamma=args.gamma,
        recon_weight=args.recon_weight
    )
    
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
