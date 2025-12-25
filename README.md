# Fine-tuning DINOv3 with Metric Learning for Image Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

English | [日本語](README_jp.md)

## Highlights

- 🔍 **Image retrieval using DINOv3 as backbone** - Leverage powerful Vision Transformer features
- 💪 **Batch-hard Triplet Loss** for stable metric learning under few-shot and class-imbalanced settings
- 🎨 **DINO-style augmentation** to ensure positive pairs even with limited data
- ⚡ **Mixed precision training (AMP)** for reduced GPU memory usage and larger effective batch sizes

## Overview

This project provides a training framework for metric learning on image retrieval tasks using **DINOv3** (Vision Transformer) as the backbone model. The implementation uses **Batch Hard Triplet Loss** with online mining to learn robust image embeddings suitable for similarity search and retrieval applications.

## Requirements

**Tested Environment:**
- Python 3.10
- PyTorch 2.7.1
- Ubuntu 22.04
- RTX 4060 Ti

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/lalafloat/dinov3-metric-learning.git
cd dinov3-metric-learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

For detailed system requirements, see the [DINOv3 repository](https://github.com/facebookresearch/dinov3).

### 3. Prepare Pretrained Model

The DINOv3 pretrained model will **NOT** be automatically downloaded because `train_model.py` uses `local_files_only=True` to enforce the use of cached models. You must manually download the model to the `weights/` directory:

```python
from transformers import AutoModel

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
model = AutoModel.from_pretrained(model_name, device_map="auto", cache_dir="weights")
```

For more details on available pretrained models, see the [DINOv3 Hugging Face documentation](https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-hugging-face-transformers).

### 4. Prepare Dataset

Organize your dataset with the following directory structure:

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

**Notes:**
- Each class should have its own subdirectory
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Images will be automatically resized and augmented during training

## Usage

### Basic Training

```bash
python train.py output/
```

This will:
- Use default configuration from `config.py`
- Save checkpoints to `output/checkpoints/`
- Write training logs to `output/log.txt`

### Training with Custom Settings

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

## Configuration

### Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Hugging Face model name |
| `model_dir` | `weights` | Directory to cache pretrained model |
| `proj_dim` | `384` | Dimension of projection head output |
| `freeze_until` | `10` | Freeze encoder layers with index < freeze_until |

### Data Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_dir` | `data/train` | Training data directory |
| `valid_dir` | `data/valid` | Validation data directory |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `32` | Batch size |
| `num_workers` | `4` | Number of data loading workers |
| `learning_rate` | `1e-5` | Learning rate |
| `weight_decay` | `0.01` | Weight decay |
| `num_epochs` | `10` | Number of training epochs |
| `margin` | `0.2` | Triplet loss margin |
| `grad_clip` | `1.0` | Gradient clipping threshold |

### Batch Hard Triplet Loss

For each anchor in the batch:
- **Hardest positive**: Farthest same-class sample
- **Hardest negative**: Closest different-class sample
- **Loss**: `max(0, margin + d(anchor, hardest_pos) - d(anchor, hardest_neg))`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dinov3-metric-learning,
  title={DINOv3 Metric Learning},
  author={lalafloat},
  year={2025},
  url={https://github.com/lalafloat/dinov3-metric-learning}
}
```

## Acknowledgments

- [DINOv3](https://github.com/facebookresearch/dinov3) - Meta AI Research
