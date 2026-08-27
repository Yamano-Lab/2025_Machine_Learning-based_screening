# English

# Machine Learning Screening / Pyrenoid Morphology Analysis

Author: Kojiro Matsuo
Date created: December 22, 2025

## 1. File Structure

```text
ML_based_screening/
├── data/                     # Storage for training and analysis data
│   └── raw_images/           # Raw image data
├── models/                   # Trained model storage
├── results/                  # Training result output directory
│   └── vae_vgg_output/       # VAE training results, etc.
├── screening_results/        # Screening analysis result output directory
├── src/                      # Python source code (core logic)
│   ├── __init__.py           # Package initialization (normally does not need editing)
│   ├── CAE_improved_modeltrain.py  # CAE model training
│   ├── train_vae_vgg.py      # High-sensitivity VAE model training
│   ├── train_aligned_cae.py  # CAE training with aligned images
│   ├── train_dual_channel_cae.py # Two-channel input CAE training
│   ├── integrated_screening.py     # Screening analysis using a trained model
│   ├── integrated_screening_aligned.py # Screening analysis of aligned images
│   ├── create_aligned_dataset_v5.py # Dataset alignment and preprocessing
│   ├── visualize_average_error.py  # Visualization of mean reconstruction error
│   ├── refine_anomaly_detector.py  # Retuning anomaly detectors (e.g., OneClassSVM)
│   ├── inspect_dataset.py          # Dataset inspection and visualization
│   └── generate_image_only.py      # Image-generation helper
├── Scripts/                  # Shell scripts for execution (intended for user editing)
│   ├── align.sh              # Image alignment script
│   ├── run_train.sh          # Training script
│   ├── run_train_aligned.sh  # Training script for aligned images
│   ├── run_train_dual.sh     # Dual-channel CAE training script
│   ├── run_vae_training.sh   # VAE training script
│   └── run_analysis.sh       # Analysis script
├── requirements.txt          # List of required libraries
├── .gitignore                # Specifies files excluded from Git
└── README.md                 # Documentation (this file)
```

## 2. Environment Setup

Please use Anaconda.

```bash
conda create -n lab_env python=3.9
conda activate lab_env
pip install -r requirements.txt
```

## 3. Usage

1. Open Terminal and navigate to the repository directory.

   ```bash
   cd Documents/YamanoLab_Scripts_Github/ML_based_screening
   ```

2. Activate the environment (confirm that `(base)` changes to `(lab_env)`).

   ```bash
   conda activate lab_env
   ```

3. Run the program.

   Before running a script, update the path settings in each file under `Scripts/`.

   **Preprocessing**

   Aligns the images and creates the dataset.

   ```bash
   bash Scripts/align.sh
   ```

   **To run training:**

   - **Standard Training (CAE)**

     Use this standard method to detect major morphological changes, such as shape deformation.

     ```bash
     bash Scripts/run_train.sh
     ```

   - **Aligned CAE Training**

     Trains the model using an aligned dataset.

     ```bash
     bash Scripts/run_train_aligned.sh
     ```

   - **Dual-Channel Training**

     Trains the model using two channels (e.g., fluorescence and bright-field images).

     ```bash
     bash Scripts/run_train_dual.sh
     ```

   - **High-Sensitivity Training (VAE + VGG)**

     This high-sensitivity method is recommended for detecting subtle differences in texture or internal structure.

     ```bash
     bash Scripts/run_vae_training.sh
     ```

   **To run analysis:**

   ```bash
   bash Scripts/run_analysis.sh
   ```

---

## 4. Git Update Instructions for macOS (Git Cheat Sheet)

### Step 0: Setup (first time only)

```bash
# Navigate to any directory, such as the Desktop
cd ~/Desktop

# Clone the repository
git clone https://github.com/[username]/ML_based_screening.git

# Enter the repository directory
cd ML_based_screening
```

### Step 1: Update Before Starting Work

```bash
git pull origin main
```

### Step 2: Save Your Changes After Working

```bash
git add .
# Describe your changes in the commit message.
git commit -m "Describe your changes"
git push origin main
```

---

# 日本語版

# 機械学習スクリーニング / ピレノイドの形態解析

作成者: 松尾光治良
作成日: 2025/12/22

## 1. ファイル構成
```text
ML_based_screening/
├── data/                     # 学習・解析用データ置き場
│   └── raw_images/           # 原画像データ
├── models/                   # 学習済みモデルの保存先
├── results/                  # 学習結果の出力先
│   └── vae_vgg_output/       # VAE学習の結果など
├── screening_results/        # スクリーニング解析結果の出力先
├── src/                      # Pythonのソースコード（コアロジック）
│   ├── __init__.py           # パッケージ認識用（通常は触りません）
│   ├── CAE_improved_modeltrain.py  # CAEモデルの学習
│   ├── train_vae_vgg.py      # 高感度VAEモデルの学習
│   ├── train_aligned_cae.py  # 位置合わせ済み画像のCAE学習
│   ├── train_dual_channel_cae.py # 2チャンネル入力CAE学習
│   ├── integrated_screening.py     # 学習済みモデルを用いたスクリーニング解析
│   ├── integrated_screening_aligned.py # 位置合わせ済み画像のスクリーニング解析
│   ├── create_aligned_dataset_v5.py # データセットの位置合わせ・前処理
│   ├── visualize_average_error.py  # 平均再構成誤差の可視化
│   ├── refine_anomaly_detector.py  # 異常検知器(OneClassSVM等)の再調整
│   ├── inspect_dataset.py          # データセットの検査・可視化
│   └── generate_image_only.py      # 画像生成用ヘルパー
├── Scripts/                  # 実行用のシェルスクリプト（ユーザーが触る場所）
│   ├── align.sh              # 画像の位置合わせ実行用
│   ├── run_train.sh          # 学習実行用
│   ├── run_train_aligned.sh  # 位置合わせ済み画像の学習実行用
│   ├── run_train_dual.sh     # Dual Channel CAE学習実行用
│   ├── run_vae_training.sh   # VAE学習実行用
│   └── run_analysis.sh       # 解析実行用
├── requirements.txt          # 必要なライブラリ一覧
├── .gitignore                # Gitにあげないファイル指定
└── README.md                 # 説明書（このファイル）
```

## 2. 環境構築
Anacondaを使用してください。

```bash
conda create -n lab_env python=3.9
conda activate lab_env
pip install -r requirements.txt
```

## 3. 実行方法 (Usage)
1. Terminalを開き、リポジトリの場所に移動する。
    ```bash
    cd Documents/YamanoLab_Scripts_Github/ML_based_screening
    ```
2. 環境を有効化する（`(base)` から `(lab_env)` に変わったことを確認）。
    ```bash
    conda activate lab_env
    ```
3. プログラムを実行する。
    ※実行前に `Scripts/` 内の各ファイルのパス設定を変更してください。

    **前処理 (Preprocessing):**
    画像の位置合わせとデータセット作成を行います。
    ```bash
    bash Scripts/align.sh
    ```

    **学習を実行する場合:**
    
    *   **通常学習 (CAE):**
        （標準）形状の崩れなど、大きな形態変化を検出する場合に使用します。
        ```bash
        bash Scripts/run_train.sh
        ```
    
    *   **位置合わせ済み学習 (Aligned CAE):**
        位置合わせ済みのデータセットを用いて学習します。
        ```bash
        bash Scripts/run_train_aligned.sh
        ```

    *   **Dual Channel 学習:**
        2つのチャンネル（例: 蛍光と明視野）を用いて学習します。
        ```bash
        bash Scripts/run_train_dual.sh
        ```
    
    *   **高感度学習 (VAE + VGG):**
        （高感度）微細なテクスチャや内部構造の違いを検出する場合に使用します（推奨）。
        ```bash
        bash Scripts/run_vae_training.sh
        ```

    **解析を実行する場合:**
    ```bash
    bash Scripts/run_analysis.sh
    ```

---

## 4. 【Mac版】Git更新手順 (Git Cheat Sheet)
### ステップ0: 準備（初回のみ）
```bash
# デスクトップなど、任意の場所に移動
cd ~/Desktop

# リポジトリをダウンロード
git clone https://github.com/[ユーザー名]/ML_based_screening.git

# フォルダの中に入る
cd ML_based_screening
```

### ステップ1: 作業前の更新
```bash
git pull origin main
```

### ステップ2: 作業後の保存
```bash
git add .
git commit -m "変更内容"
git push origin main
```
