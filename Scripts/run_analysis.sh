#!/bin/bash

# --- Configuration ---

# PLEASE EDIT THE FOLLOWING PATHS BEFORE RUNNING
# ==============================================

# Path to the trained model directory (containing .keras files)
MODEL_DIR="/path/to/your/trained_model_directory"

# Root directory containing subfolders of screening data (Mutants/Conditions)
INPUT_DIR="/path/to/your/screening_data_root"

# Path to WT (Wild Type) data for baseline comparison (file or folder)
WT_PATH="/path/to/your/WT_data"

# Output directory for results
OUTPUT_DIR="/path/to/your/results_directory"

# ---------------------

# integrated_screening.py を実行します。
# 各オプションの詳細は integrated_screening.py のヘルプを参照してください。
python integrated_screening.py \
  --mode umap \
  --model_dir "$MODEL_DIR" \
  --input_paths "$INPUT_DIR" \
  --wt_path "$WT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --extra_viz \
  --quantitative

# 解析完了メッセージ
echo "解析が完了しました。結果は $OUTPUT_DIR を確認してください。"
