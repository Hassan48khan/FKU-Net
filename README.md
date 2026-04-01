# FKU-Net: A Fourier-KAN Driven U-Net with Boundary-Aware Gated Progressive Fusion for Cardiac Image Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Conference-IEEE-blue" alt="IEEE"/>
</p>

> **FKU-Net: A Fourier-KAN Driven U-Net with Boundary-Aware Gated Progressive Fusion for Cardiac Image Segmentation**  
> Hassan Ali, Muhammad Asghar Khan*, Hafsa Waheed, Marwa Anjum, Farhan Ullah  
> *\* Corresponding author — Prince Mohammad Bin Fahd University*

**Code:** [https://github.com/Hassan48khan/FKU-Net](https://github.com/Hassan48khan/FKU-Net)

---

## Overview

FKU-Net is a Fourier–Kolmogorov–Arnold enhanced U-Net for accurate cardiac image segmentation. It addresses two key limitations of existing KAN-based segmentation models: limited adaptability to diverse imaging conditions, and insufficient boundary precision in low-contrast ultrasound and MRI data.

**Core idea:** Replace B-spline KAN activations with learnable Fourier-series functions (FourierKAN), which are inherently periodic, numerically stable, and more expressive for frequency-domain patterns. Pair this with a novel boundary-guided skip-connection mechanism (BAGPF) that uses prior decoder predictions to actively refine encoder features before fusion.

---

## Architecture

```
Input (1×H×W)
    │
    ▼
┌──────────────────────────────────────┐
│  Encoder  (3 × MSRCA + MaxPool×2)   │
│  ResConv(3×3) → DWConv(3/5/7) → EMA │
│  OverlapPatchEmbed at 1/8 & 1/16    │
└────────────────┬─────────────────────┘
                 │  skip connections (x3)
    ┌────────────▼────────────┐
    │  Bottleneck: 2×PureFKANBlock   │
    │  PatchEmbed → EnhancedFKANBlock│
    │  FKAN(x) = Σ [cos(kx)·a + sin(kx)·b] │
    └────────────┬────────────┘
                 │
┌────────────────▼─────────────────────┐
│  Decoder (3 × MSRCA + Upsample×2)   │
│  Skip fusion via BAGPF at each stage │
│  Stage1: Boundary-Aware Refinement  │
│  Stage2: Multi-Scale Spatial Context │
│  Stage3: Gated Adaptive Fusion       │
└────────────────┬─────────────────────┘
                 │
    ┌────────────▼────────────┐
    │  Deep Supervision Heads │
    │  + Final Conv1×1 Output │
    └─────────────────────────┘
```

### Core Components

| Component | Role |
|---|---|
| **MSRCA** | Multi-Scale Residual Convolution Attention — residual conv + parallel DWConv (3/5/7) + EMA |
| **PureFKANBlock** | Fourier-KAN-only bottleneck (no conv branches) for frequency-aware nonlinear modeling |
| **BAGPF** | Boundary-Aware Gated Progressive Fusion — prediction-guided skip connection refinement |
| **EMA** | Efficient Multi-Scale Attention — grouped channel reweighting with anisotropic spatial context |
| **Deep Supervision** | Auxiliary prediction heads at intermediate decoder stages |

---

## Results

### ACDC Cardiac MRI Dataset

| Model | Mean Dice (%) | HD95 (mm) | Mean IoU (%) | RV (%) | Myo (%) | LV (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| UNet | 87.55 | 8.12 | 78.34 | 87.10 | 80.63 | 94.92 |
| TransUNet | 89.71 | 2.54 | 82.46 | 88.86 | 84.53 | 95.73 |
| Swin-UNet | 90.00 | 4.52 | 83.21 | 88.55 | 85.62 | 95.83 |
| MT-UNet | 90.43 | 1.20 | 83.85 | 86.64 | 89.04 | 95.62 |
| UKAN | 91.30 | 1.22 | 84.67 | 91.37 | 90.14 | 95.39 |
| **FKU-Net (Ours)** | **92.42** | **1.09** | **85.76** | **91.98** | **90.21** | **96.20** |

### CAMUS 2D Echocardiography — Mask 1 (LV Endocardium)

| Model | 2CH-ED Dice | 2CH-ES Dice | 4CH-ED Dice | 4CH-ES Dice |
|---|:---:|:---:|:---:|:---:|
| UNet | 0.9156 | 0.8953 | 0.9101 | 0.9130 |
| TransUNet | 0.9284 | 0.8990 | 0.9455 | 0.9221 |
| Swin-UNet | 0.9363 | 0.9157 | 0.9483 | 0.9276 |
| U-KAN | 0.9299 | 0.9020 | 0.9394 | 0.9286 |
| **FKU-Net** | **0.9421** | **0.9241** | **0.9518** | **0.9328** |

### CAMUS — Mask 2 (Myocardium, 2CH-ED)

| Model | Dice | IoU | HD (mm) |
|---|:---:|:---:|:---:|
| UNet++ | 0.8193 | 0.7002 | 25.25 |
| U-KAN | 0.8525 | 0.7458 | 6.14 |
| **FKU-Net** | **0.8675** | **0.7680** | **4.98** |

### CAMUS — Mask 3 (Left Atrium, 4CH-ES)

| Model | Dice | IoU | HD (mm) |
|---|:---:|:---:|:---:|
| Swin-UNet | 0.9184 | 0.8609 | 8.92 |
| U-KAN | 0.9326 | 0.8751 | 4.26 |
| **FKU-Net** | **0.9416** | **0.8865** | **3.68** |

---

## Repository Structure

```
FKU-Net/
├── fku_net.py            # Main FKU-Net model
├── msrca.py              # MSRCA block (Multi-Scale Residual Convolution Attention)
├── bagpf.py              # BAGPF skip-fusion module
├── fourier_kan.py        # FourierKAN layer (FKAN) and PureFKANBlock
├── ema.py                # Efficient Multi-Scale Attention module
├── train.py              # Training script (ACDC / CAMUS)
├── test.py               # Evaluation script
├── datasets/
│   ├── acdc.py           # ACDC dataloader
│   └── camus.py          # CAMUS dataloader
├── utils/
│   ├── losses.py         # Dice + Cross-Entropy loss
│   └── metrics.py        # DSC, IoU, HD95
└── README.md
```

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (tested with PyTorch 2.x + CUDA 11.8)
- `timm`, `numpy`, `scipy`, `medpy` (for HD95)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm numpy scipy medpy
```

### Clone

```bash
git clone https://github.com/Hassan48khan/FKU-Net.git
cd FKU-Net
```

---

## Quick Start

### Instantiate FKU-Net

```python
import torch
from fku_net import FKUNet

model = FKUNet(
    num_classes=3,        # 3 for ACDC (LV, RV, Myo); 1 for binary CAMUS masks
    input_channels=1,     # grayscale
    img_size=128,
    embed_dims=[64, 128, 256],
    fkan_grid_size=5,     # Fourier grid size g
    fkan_depth=3,         # number of EnhancedFKANBlock layers in bottleneck
).cuda()

x = torch.randn(2, 1, 128, 128).cuda()
out = model(x)            # shape: (2, num_classes, 128, 128)
print(out.shape)
```

### FourierKAN Layer (standalone)

```python
from fourier_kan import FourierKANLayer

# Replaces a standard linear layer with Fourier-basis activation
layer = FourierKANLayer(in_features=256, out_features=256, grid_size=5)
x = torch.randn(16, 256)
y = layer(x)   # (16, 256)
```

The Fourier activation per input-output pair is:

```
φ_F(x_i) = Σ_{k=1}^{g}  [ cos(k·x_i)·a_{ik}  +  sin(k·x_i)·b_{ik} ]
```

where `a`, `b` are learnable and `g` is the grid size.

---

## Training

### ACDC

```bash
python train.py \
  --dataset acdc \
  --data_dir /path/to/ACDC \
  --num_classes 3 \
  --img_size 128 \
  --batch_size 4 \
  --epochs 100 \
  --optimizer adam \
  --lr 1e-3 \
  --loss weighted_dice_ce \
  --aug_prob 0.5
```

### CAMUS

```bash
python train.py \
  --dataset camus \
  --data_dir /path/to/CAMUS \
  --num_classes 1 \
  --img_size 128 \
  --batch_size 4 \
  --epochs 50 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --kfold 10
```

### Training Settings Summary

| Dataset | Optimizer | LR | Epochs | Batch | Augmentation | Loss |
|---|---|---|---|---|---|---|
| ACDC | Adam | 1e-3 | 100 | 4 | Rotation ±, H/V flip (p=0.5) | Weighted Dice + CE |
| CAMUS | SGD (mom=0.9) | 0.01 | 50 | 4 | None (strict protocol) | Dice + CE |

All images resized to **128×128**. GPU: any with ≥8 GB VRAM.

---

## Datasets

| Dataset | Modality | Subjects | Structures | Split | Access |
|---|---|---|---|---|---|
| [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | 2D Echo | 500 | LVendo, LVepi, LA | 450 train / 50 test (10-fold CV) | Public |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cardiac MRI | 150 (100 used) | LV, RV, Myo | 70 / 10 / 20 | Public |

**CAMUS:** 10-fold cross-validation, patient-disjoint, images resized from 778×594 → 128×128, **no data augmentation**.  
**ACDC:** Standard split following TransUNet / MT-UNet protocol, results averaged over 5 runs.

---

## Module Details

### MSRCA — Multi-Scale Residual Convolution Attention

Each encoder/decoder stage uses MSRCA to extract rich multi-scale features:

```
Input X
  ├─ ResConv branch: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU  (+ shortcut)
  └─ Multi-scale branch:
       DWConv3×3(X_main) ┐
       DWConv5×5(X_main) ├─ Concat → GroupBN → EMA → Conv1×1
       DWConv7×7(X_main) ┘
          └── + ResConv output → X_out
```

The parallel depthwise kernels capture fine (k=3), medium (k=5), and coarse (k=7) cardiac structures simultaneously, while EMA reweights channels based on both directional pooling and local 3×3 texture.

### BAGPF — Boundary-Aware Gated Progressive Fusion

Applied at every skip connection in the decoder, using the prior decoder prediction `p` to guide refinement:

**Stage 1 — Boundary-Aware Refinement:**
```
fg_attn  = σ(p)                              # foreground attention
bd_attn  = clamp(1 - |σ(p) - 0.5| / 0.5)   # uncertainty / boundary map
x_l_ref  = Conv1×1(SE(cat(Conv_fg(x_l ⊙ fg_attn),
                           Conv_bd(x_l ⊙ bd_attn)))) + x_l
```

**Stage 2 — Multi-Scale Spatial Context:**
```
cat = [x_l_refined, upsample(proj(x_h))]
dilated = cat([DWConv_d(group_i) for d in {1,3,5,7}])
```

**Stage 3 — Gated Adaptive Fusion:**
```
A       = CA(dilated) ⊙ SA(dilated)          # channel + spatial attention
G       = σ(BN(Conv1×1(cat)))               # gating map
F_fused = G ⊙ (x_h ⊙ A) + (1-G) ⊙ (x_l_refined ⊙ A)
```

### PureFKANBlock — Fourier-KAN Bottleneck

Replaces the standard ASPP or KAN bottleneck with a pure Fourier-KAN pipeline:

```
X → PatchEmbed (Conv3×3 + LN + flatten) → [EnhancedFKANBlock] × depth
    where each block: Z = LN(X + DWConv(FKAN(X)))
```

Uses only Fourier activations — no convolutional branches — for maximum interpretability and frequency-domain expressiveness. The 3-layer variant (depth=3) is optimal per ablation.

---

## Ablation Study (ACDC)

| Configuration | Params (M) | GFLOPs | Dice (%) | IoU (%) |
|---|:---:|:---:|:---:|:---:|
| Baseline (MSRCA encoder/decoder only) | 8.4 | 3.2 | 87.6 | 78.5 |
| + 1-layer standard KAN | 8.9 | 3.5 | 89.1 | 80.8 |
| + 3-layer standard KAN | 9.5 | 3.9 | 90.8 | 83.4 |
| + 3-layer Fourier-KAN (FKAN) | 9.3 | 3.8 | 91.9 | 85.0 |
| FKU-Net w/o MSRCA | 9.7 | 4.0 | 91.2 | 84.1 |
| FKU-Net w/o BAGPF | 9.8 | 4.1 | 91.3 | 84.3 |
| **FKU-Net (full)** | **10.0** | **4.2** | **92.4** | **85.8** |

Key findings:
- FKAN outperforms standard KAN by **+1.1% Dice** at *lower* parameter count (9.3M vs 9.5M)
- 5-layer KAN adds only +0.2% Dice over 3-layer, confirming **depth=3 is optimal**
- MSRCA contributes **+1.2% Dice**; BAGPF contributes **+1.1% Dice** independently
- All three components together yield synergistic gains

---

## Key Differences: FKU-Net vs KMS-Net

Both models target cardiac segmentation and share the same research group. Here is how they differ:

| Aspect | FKU-Net | KMS-Net |
|---|---|---|
| KAN type | **FourierKAN** (Fourier coefficients) | Standard KANLinear (B-splines) |
| Bottleneck | PureFKANBlock (no conv branches) | KASPPS (KAN atrous conv + SE) |
| Skip fusion | **BAGPF** (prediction-guided, 3-stage) | MSAG (multi-receptive-field gates) |
| Encoder | MSRCA (DWConv 3/5/7 + EMA) | ResConv + EMA |
| Global modeling | Deep supervision heads | SS2D (state-space, linear complexity) |
| Parameters | ~10M | ~32.8M |
| Primary advantage | Lightweight, frequency-aware, interpretable | High-capacity, strong cross-modal generalization |

## Citation

If you use FKU-Net in your research, please cite:

```bibtex
@inproceedings{ali2025fkunet,
  author    = {Hassan Ali and Muhammad Asghar Khan and Hafsa Waheed
               and Marwa Anjum and Farhan Ullah},
  title     = {{FKU-Net}: A {Fourier-KAN} Driven {U-Net} with
               Boundary-Aware Gated Progressive Fusion for
               Cardiac Image Segmentation},
  booktitle = {Proceedings of the IEEE International Conference},
  year      = {2025}
}
```

---

## Credits

- FourierKAN implementation inspired by [GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) (Xu et al., 2024)
- EMA module from [Efficient Multi-Scale Attention](https://arxiv.org/abs/2305.13563) (Ouyang et al., ICASSP 2023)
- U-KAN baseline from [U-KAN](https://arxiv.org/abs/2406.02918) (Li et al., AAAI 2025)
- CAMUS dataset: [Leclerc et al., IEEE TMI 2019](https://doi.org/10.1109/TMI.2019.2900516)
- ACDC dataset: [Bernard et al., IEEE TMI 2018](https://doi.org/10.1109/TMI.2018.2837502)

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

- **Hassan Ali** — ali.hassan@nuaa.edu.cn (Nanjing University of Aeronautics and Astronautics)
- **Muhammad Asghar Khan** *(corresponding)* — mkhan4@pmu.edu.sa (Prince Mohammad Bin Fahd University)
