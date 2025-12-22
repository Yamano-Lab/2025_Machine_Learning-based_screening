#!/bin/bash

# ==========================================
# Training Configuration
# ==========================================

# PLEASE EDIT THE FOLLOWING PATHS BEFORE RUNNING
# ==============================================

# Input directory containing training images (WT)
# Example: "/Users/username/data/WT_images"
INPUT_DIR="/path/to/your/training_data_WT"

# Output directory for saving models
# Example: "/Users/username/data/models"
# A timestamped subfolder will be created inside this directory
OUTPUT_BASE_DIR="/path/to/your/model_output_dir"
DATE_STR=$(date "+%Y%m%d_%H%M")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATE_STR}"

# ==========================================
# Execution
# ==========================================

echo "Starting Autoencoder Training..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

# Pythonスクリプトの実行
python CAE_improved_modeltrain.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

# エラーハンドリング
if [ $? -eq 0 ]; then
    echo "Training finished successfully!"
else
    echo "Training failed!"
    exit 1
fi