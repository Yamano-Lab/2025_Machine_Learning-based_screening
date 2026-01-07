#!/bin/bash

# Default directories
INPUT_DIR="data/aligned_dataset_v3"
OUTPUT_DIR="results/Models/Aligned_$(date +%Y%m%d)"

# Override if arguments provided
if [ "$1" != "" ]; then
    INPUT_DIR="$1"
fi

if [ "$2" != "" ]; then
    OUTPUT_DIR="$2"
fi

echo "Starting Aligned CAE Training..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python src/train_aligned_cae.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

echo "Training script finished."
