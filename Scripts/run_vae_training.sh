#!/bin/bash

# Default parameters
DATA_DIR="./data/raw_images"  # Change this to your actual data path
OUTPUT_DIR="./results/vae_vgg_output"
EPOCHS=100
BATCH_SIZE=32
BETA=1.0    # Weight for KL Divergence
GAMMA=0.0002   # Weight for VGG Perceptual Loss
RECON_WEIGHT=100.0 # Weight for Reconstruction Loss (MAE)

# Help function
show_help() {
    echo "Usage: ./run_vae_training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data_dir DIR      Path to input TIFF files (default: $DATA_DIR)"
    echo "  -o, --output_dir DIR    Path to save outputs (default: $OUTPUT_DIR)"
    echo "  -e, --epochs INT        Number of epochs (default: $EPOCHS)"
    echo "  -b, --batch_size INT    Batch size (default: $BATCH_SIZE)"
    echo "  --beta FLOAT            Weight for KL Divergence (default: $BETA)"
    echo "  --gamma FLOAT           Weight for VGG Perceptual Loss (default: $GAMMA)"
    echo "  --recon_weight FLOAT    Weight for Reconstruction Loss (default: $RECON_WEIGHT)"
    echo "  -h, --help              Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--data_dir)
        DATA_DIR="$2"
        shift 2
        ;;
        -o|--output_dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
        -e|--epochs)
        EPOCHS="$2"
        shift 2
        ;;
        -b|--batch_size)
        BATCH_SIZE="$2"
        shift 2
        ;;
        --beta)
        BETA="$2"
        shift 2
        ;;
        --gamma)
        GAMMA="$2"
        shift 2
        ;;
        --recon_weight)
        RECON_WEIGHT="$2"
        shift 2
        ;;
        -h|--help)
        show_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
done

# Get the directory of this script to locate the python script correctly
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/src/train_vae_vgg.py"

echo "========================================================"
echo "Starting VAE Training with VGG Perceptual Loss"
echo "========================================================"
echo "Date: $(date)"
echo "Python Script: $PYTHON_SCRIPT"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Beta (KL Weight): $BETA"
echo "Gamma (VGG Weight): $GAMMA"
echo "Recon Weight: $RECON_WEIGHT"
echo "========================================================"

# Check if python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the python script
python "$PYTHON_SCRIPT" \
    --input_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --beta $BETA \
    --gamma $GAMMA \
    --recon_weight $RECON_WEIGHT

# Check exit code
if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Training completed successfully."
    echo "Results saved to: $OUTPUT_DIR"
    echo "========================================================"
else
    echo "========================================================"
    echo "Error: Training failed."
    echo "========================================================"
    exit 1
fi
