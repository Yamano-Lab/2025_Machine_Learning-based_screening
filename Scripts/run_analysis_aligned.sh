#!/bin/bash

# --- 設定項目 ---

# 学習済みモデルのディレクトリ（Alignedモデルを指定）
MODEL_DIR="results/Models/Aligned_20260115"

# 解析対象のサンプル群が入っているルートフォルダのパス
# 例: "data/raw_images/Mutant_Pool" など
INPUT_DIR="data/aligned_dataset_v5/Candidates_01"

# WT（野生型）のデータが入っているフォルダのパス
# ベースライン計算に使用します
WT_PATH="data/aligned_dataset_v5/WT"

# 出力先ディレクトリ（指定しない場合は自動生成されます）
OUTPUT_DIR="results/analysis_results/Aligned_Analysis_$(date +%Y%m%d_%H%M)"

# --- 解析の実行 ---

# 自動モード判定: INPUT_DIR 内にサブディレクトリがあるか確認
if [ -d "$INPUT_DIR" ] && [ -n "$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -not -name '.*')" ]; then
    echo "サブディレクトリを検出しました。FOLDER (UMAP) モードで実行します。"
    DETECTED_MODE="umap"
else
    echo "サブディレクトリが検出されませんでした。FILE モードで実行します。"
    DETECTED_MODE="file"
fi

echo "開始: Aligned Screening Pipeline"
echo "Model: $MODEL_DIR"
echo "Input: $INPUT_DIR"
echo "WT:    $WT_PATH"
echo "Output: $OUTPUT_DIR"

# integrated_screening_aligned.py を実行
# --use_prealigned: 入力が既に位置合わせ済み(64x64x2)の場合に指定
# 今回は aligned_dataset_v5 を入力としているため、use_prealigned を有効化しても良いですが、
# スクリプト内の自動判別に任せるか、明示的に指定します。
# もし入力が raw_images なら --use_prealigned は外してください。

python src/integrated_screening_aligned_v2.py \
  --mode "$DETECTED_MODE" \
  --umap \
  --model_dir "$MODEL_DIR" \
  --input_paths "$INPUT_DIR" \
  --wt_path "$WT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --extra_viz \
  --quantitative \
  --use_prealigned

echo "解析が完了しました。結果は $OUTPUT_DIR を確認してください。"
