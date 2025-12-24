# [機械学習スクリーニング / ピレノイドの形態解析]

作成者: 松尾光治良
作成日: 2025/12/22

## 1. ファイル構成
```text
ML_based_screening/
├── src/                      # Pythonのソースコード（コアロジック）
│   ├── __init__.py           # パッケージ認識用（通常は触りません）
│   ├── CAE_improved_modeltrain.py  # CAEモデルの学習
│   ├── train_vae_vgg.py      # 高感度VAEモデルの学習
│   ├── integrated_screening.py     # 学習済みモデルを用いたスクリーニング解析
│   └── generate_image_only.py      # 画像生成用ヘルパー
├── Scripts/                  # 実行用のシェルスクリプト（ユーザーが触る場所）
│   ├── run_train.sh          # 学習実行用
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

    **学習を実行する場合:**
    
    *   **通常学習 (CAE):**
        （標準）形状の崩れなど、大きな形態変化を検出する場合に使用します。
        ```bash
        bash Scripts/run_train.sh
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