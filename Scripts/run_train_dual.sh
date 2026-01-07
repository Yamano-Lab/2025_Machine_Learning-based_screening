#!/bin/bash

# Default directories
INPUT_DIR="data/aligned_images"
OUTPUT_DIR="results/vae_vgg_output"

# Override if arguments provided
if [ "$1" != "" ]; then
    INPUT_DIR="$1"
fi

if [ "$2" != "" ]; then
    OUTPUT_DIR="$2"
fi

echo "Starting Dual Channel CAE Training..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python src/train_dual_channel_cae.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

echo "Training script finished."