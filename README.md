<div align="center">

# <img src="assets/gmff_logo.svg" alt="GMFF" height="800" style="vertical-align: middle;"/> GMFF

**Generative Multi-focus Image Fusion Network**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-GMFF-black.svg)](https://github.com/Xinzhe99/GMFF)

*Official PyTorch implementation for Generative Multi-focus Image Fusion*

</div>

## 📢 News

> [!NOTE]
> 🎉 **2025.11**: The paper **Generative Multi-focus Image Fusion Network** has been submitted.

## Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Downloads](#-downloads)
- [Usage](#-usage)
- [Training](#-training)
- [Citation](#-citation)

## 📖 Overview

<div align="center">
<img src="assets/gmff_framework.jpg" width="800px"/>
</div>

## ✨ Highlights

- Presents the first generative multi-focus image fusion network based on diffusion models.
- Combines the strengths of stack-based fusion and generative modeling for enhanced results.
- Employs a two-stage pipeline: stack fusion followed by diffusion-based refinement.
- Leverages pre-trained stable diffusion models for high-quality image generation.
- Provides an open-source solution that outperforms existing methods with superior visual quality.

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Xinzhe99/GMFF.git
cd GMFF
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n gmff python=3.8
conda activate gmff
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📥 Downloads

| Resource | Link | Code | Description |
|----------|------|------|-------------|
| 🗂️ **Test Datasets** | [![Download](https://img.shields.io/badge/Download-4CAF50?style=flat-square)](https://pan.baidu.com/s/1XrKGlqSK6kc_R-1AzprHlA?pwd=cite) | `cite` | Complete test datasets |
| 📊 **Benchmark Results** | [![Download](https://img.shields.io/badge/Download-FF9800?style=flat-square)](https://pan.baidu.com/s/1_rBtM9o7RUQP4oyt8HHXwg?pwd=cite) | `cite` | Fusion results from all methods |
| 🔧 **Pre-trained Models** | [![Download](https://img.shields.io/badge/Download-2196F3?style=flat-square)](https://pan.baidu.com/s/1example) | `gmff` | Pre-trained GMFF models |

## 💻 Usage

### Stage 1: Stack-based Fusion

The pre-trained StackMFF V4 model weights file (`stackmffv4.pth`) should be placed in the [weights](weights/) directory.

To fuse a stack of multi-focus images, organize your input images in a folder with numeric filenames (e.g., `0.png`, `1.png`, etc.):

```
input_stack/
├── 0.png
├── 1.png
├── 2.png
└── 3.png
```

Run the Stage 1 prediction script:

```bash
python inference_stage1.py --input_dir ./input_stack --output_dir ./results_stage1
```

### Stage 2: Diffusion-based Refinement

To refine the fused results using the diffusion model, run:

```bash
python inference_stage2.py --input_dir ./results_stage1 --output_dir ./results_stage2
```

### Batch Processing

To perform batch processing on multiple test datasets, organize your data in the following directory structure:

```
test_datasets/
├── Dataset1/
│   └── TR/
│       └── focus_stack/
│           ├── scene1/
│           │   ├── 0.png
│           │   ├── 1.png
│           │   └── 2.png
│           └── scene2/
│               ├── 0.png
│               ├── 1.png
│               └── 2.png
├── Dataset2/
│   └── TR/
│       └── focus_stack/
└── Dataset3/
    └── TR/
        └── focus_stack/
```

Run the Stage 1 batch processing script:

```bash
python datasets/step2_make_datasets_for_gmff.py --test_root ./test_datasets --test_datasets Dataset1 Dataset2 Dataset3
```

## 🏋️ Training

### Dataset Structure

The GMFF training pipeline consists of two stages:

#### Stage 1 - StackMFF V4 Training:

```
stackmff_datasets/
├── DatasetName1/
│   ├── TR/ (Training set)
│   │   ├── focus_stack/ (image stacks)
│   │   │   ├── scene1/
│   │   │   │   ├── 0.png
│   │   │   │   ├── 1.png
│   │   │   │   └── 2.png
│   │   │   └── scene2/
│   │   │       ├── 0.png
│   │   │       ├── 1.png
│   │   │       └── 2.png
│   │   └── focus_index_gt/ (Focus index ground truth)
│   │       ├── scene1.npy
│   │       └── scene2.npy
│   └── TE/ (Test/Validation set)
│       ├── focus_stack/
│       └── focus_index_gt/
├── DatasetName2/
│   ├── TR/
│   └── TE/
└── ...
```

#### Stage 2 - GMFF Training:

```
gmff_datasets/
├── DatasetName1/
│   ├── TR/ (Training set)
│   │   ├── focus_stack/ (image stacks)
│   │   ├── AiF/ (All-in-Focus ground truth)
│   │   └── AiF_missing/ (Stage 1 fusion results)
│   └── TE/ (Test/Validation set)
│       ├── focus_stack/
│       ├── AiF/
│       └── AiF_missing/
├── DatasetName2/
│   ├── TR/
│   └── TE/
└── ...
```

### Training Stage 1

To train the StackMFF V4 model (Stage 1), run the following command:

```bash
python train_stage1.py \
  --save_name train_stackmffv4 \
  --datasets_root /path/to/stackmff_datasets \
  --train_datasets DatasetName1 DatasetName2 \
  --val_datasets DatasetName1 DatasetName2 \
  --batch_size 8 \
  --num_epochs 50 \
  --lr 1e-3 \
  --gpu_ids 0
```

### Training Stage 2

To train the GMFF model (Stage 2), run the following command:

```bash
python train_stage2.py \
  --config configs/train/train_stage2.yaml \
  --ckpt /path/to/pretrained/checkpoint.pt
```

## 📚 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{gmff2025,
  title={Generative Multi-focus Image Fusion},
  author={Xie, Xinzhe and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🙏 Acknowledgments

This codebase is built upon several excellent open-source projects:
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [PyTorch](https://pytorch.org/)

<div align="center">

⭐ If you find this project helpful, please give it a star and cite our paper!

</div>
