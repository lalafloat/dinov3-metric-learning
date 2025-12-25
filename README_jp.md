# Fine-tuning DINOv3 with Metric Learning for Image Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 日本語

## ハイライト

- 🔍 **DINOv3をバックボーンとした画像検索** - 強力なVision Transformer特徴量を活用
- 💪 **Batch-hard Triplet Loss** - Few-shotやクラス不均衡な設定下でも安定したMetric Learning
- 🎨 **DINOスタイルのデータ拡張** - 限られたデータでもポジティブペアを確保
- ⚡ **Mixed precision training (AMP)** - GPU メモリ使用量を削減し、より大きな実効バッチサイズを実現

## 概要

このプロジェクトは、**DINOv3**（Vision Transformer）をバックボーンモデルとして使用した画像検索タスクのためのMetric Learningフレームワークを提供します。**Batch Hard Triplet Loss**を使用して、類似度検索や検索アプリケーションに適したロバストな画像埋め込みを学習します。

## 動作環境

**動作確認環境：**
- Python 3.10
- PyTorch 2.7.1
- Ubuntu 22.04
- RTX 4060 Ti


## セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/lalafloat/dinov3-metric-learning.git
cd dinov3-metric-learning
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

詳細なシステム要件については、[DINOv3リポジトリ](https://github.com/facebookresearch/dinov3)を参照してください。

### 3. 事前学習済みモデルの準備

DINOv3の事前学習済みモデルは自動的に**ダウンロードされません**。これは`train_model.py`で`local_files_only=True`を使用してキャッシュ済みモデルの使用を強制しているためです。`weights/`ディレクトリに手動でモデルをダウンロードする必要があります：

```python
from transformers import AutoModel

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
model = AutoModel.from_pretrained(model_name, device_map="auto", cache_dir="weights")
```

利用可能な事前学習済みモデルの詳細については、[DINOv3 Hugging Face ドキュメント](https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-hugging-face-transformers)を参照してください。

### 4. データセットの準備

以下のディレクトリ構造でデータセットを整理してください：

```
data/
├── train/
│   ├── class_A/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class_B/
│   │   ├── img003.jpg
│   │   └── ...
│   └── ...
└── valid/
    ├── class_A/
    │   └── ...
    └── class_B/
        └── ...
```

**注意事項：**
- クラスごとにサブディレクトリを作る必要があります
- サポートされている画像形式：`.jpg`, `.jpeg`, `.png`, `.webp`
- 画像は学習中に自動的にリサイズおよびaugmentationされます

## 使用方法

### 基本的な学習

```bash
python train.py output/
```

これにより：
- `config.py` のデフォルト設定を使用
- チェックポイントを `output/checkpoints/` に保存
- 学習ログを `output/log.txt` に書き込み

### カスタム設定での学習

```bash
python train.py output/ \
  --train_dir data/train \
  --valid_dir data/valid \
  --model_name facebook/dinov3-vits16-pretrain-lvd1689m \
  --model_dir weights \
  --batch_size 32 \
  --num_epochs 20 \
  --learning_rate 1e-5 \
  --proj_dim 1024 \
  --freeze_until 10 \
  --margin 0.2
```

## 設定項目

### モデル設定

| パラメータ | デフォルト値 | 説明 |
|-----------|---------|-------------|
| `model_name` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Hugging Face モデル名 |
| `model_dir` | `weights` | 事前学習済みモデルをキャッシュするディレクトリ |
| `proj_dim` | `384` | プロジェクションヘッド出力の次元数 |
| `freeze_until` | `10` | インデックス < freeze_until のエンコーダレイヤーをフリーズ |

### データ設定

| パラメータ | デフォルト値 | 説明 |
|-----------|---------|-------------|
| `train_dir` | `data/train` | 学習データディレクトリ |
| `valid_dir` | `data/valid` | 検証データディレクトリ |

### 学習設定

| パラメータ | デフォルト値 | 説明 |
|-----------|---------|-------------|
| `batch_size` | `32` | バッチサイズ |
| `num_workers` | `4` | データ読み込み時のワーカー数 |
| `learning_rate` | `1e-5` | 学習率 |
| `weight_decay` | `0.01` | 重み減衰 |
| `num_epochs` | `10` | 学習エポック数 |
| `margin` | `0.2` | Triplet Loss のマージン |
| `grad_clip` | `1.0` | 勾配クリッピング閾値 |

### Batch Hard Triplet Loss

バッチ内の各アンカーに対して：
- **Hardest positive**: 最も遠い同クラスサンプル
- **Hardest negative**: 最も近い異クラスサンプル
- **Loss**: `max(0, margin + d(anchor, hardest_pos) - d(anchor, hardest_neg))`

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 引用

```bibtex
@misc{dinov3-metric-learning,
  title={DINOv3 Metric Learning},
  author={lalafloat},
  year={2025},
  url={https://github.com/lalafloat/dinov3-metric-learning}
}
```

## 謝辞

- [DINOv3](https://github.com/facebookresearch/dinov3) - Meta AI Research
