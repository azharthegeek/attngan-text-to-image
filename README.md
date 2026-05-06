# AttnGAN: Fine-Grained Text-to-Image Generation

**Group:** Mohsin Siddiqui (24K-7608) & Muhammad Azhar (24K-7606)  
**Course:** Advanced Computer Vision — NUCES  
**Base paper:** [AttnGAN: Fine-Grained Text to Image Generation with Attentional GANs](https://arxiv.org/pdf/1711.10485) (Xu et al., CVPR 2018)

---

## Overview

This project implements AttnGAN with **Spectral Normalization** added to all discriminators as an improvement over the original paper. The model generates photo-realistic bird images from free-form text descriptions at three progressively finer scales (64×64 → 128×128 → 256×256).

**Our improvement:** Spectral Normalization on all discriminator Conv2d layers enforces a Lipschitz constraint on the discriminator, stabilising adversarial training without introducing any new hyperparameters.

---

## Project Structure

```
acv-semester-project/
├── code/
│   ├── cfg/
│   │   └── bird_attn2.yml      # All hyperparameters
│   ├── datasets.py             # CUB-200-2011 data loader
│   ├── damsm.py                # Text encoder (bi-LSTM) + Image encoder (Inception-v3)
│   ├── model.py                # Generator (G0/G1/G2) + Discriminators (D0/D1/D2)
│   ├── losses.py               # GAN, DAMSM, and KL losses
│   ├── utils.py                # Checkpointing, logging, image saving
│   ├── pretrain_DAMSM.py       # Step 1: pretrain text+image encoders
│   ├── main.py                 # Step 2: train full AttnGAN
│   └── evaluate.py             # Inception Score, R-Precision, inference
├── data/birds/                 # CUB-200-2011 dataset (see setup below)
├── output/                     # Checkpoints and sample images
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Download the CUB-200-2011 dataset

```bash
# Create data directory
mkdir -p data/birds

# Download CUB-200-2011 images
# From: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# Unzip images into data/birds/images/

# Download text annotations (Reed et al. format, from StackGAN repo):
# https://github.com/hanzhanggit/StackGAN
# Place train/test split pickles and text captions under data/birds/
```

**Expected layout:**
```
data/birds/
├── images/
│   ├── 001.Black_footed_Albatross/
│   └── ...
├── text/
│   ├── 001.Black_footed_Albatross/
│   └── ...
├── train/
│   └── filenames.pickle
├── test/
│   └── filenames.pickle
└── class_info.pickle
```

---

## Training

### Step 1: Pretrain DAMSM encoders (~4–6 hours on GPU)

Trains the bi-LSTM text encoder and Inception-v3 image encoder jointly using the deep attentional multimodal similarity loss.

```bash
python code/pretrain_DAMSM.py --cfg code/cfg/bird_attn2.yml --gpu 0
```

Encoders are saved to `output/DAMSMencoders/birds/`.

### Step 2: Train AttnGAN (~12–24 hours on GPU)

Loads the pretrained encoders (frozen) and trains the multi-stage generator and discriminators adversarially.

```bash
python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0
```

To resume from a checkpoint:
```bash
python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0 \
    --resume output/checkpoints/birds/netG_epoch_0100.pth
```

Checkpoints and sample images are saved to `output/`.

---

## Inference & Evaluation

### Generate from a text description

```bash
python code/evaluate.py --cfg code/cfg/bird_attn2.yml \
    --model output/checkpoints/birds/netG_epoch_0600.pth \
    --text "this bird has a red crown with white wings and a very short beak" \
    --out output/my_bird.png --gpu 0
```

### Compute Inception Score and R-Precision

```bash
python code/evaluate.py --cfg code/cfg/bird_attn2.yml \
    --model output/checkpoints/birds/netG_epoch_0600.pth \
    --gpu 0
```

Expected output:
```
IS  : 4.36 ± 0.03   (paper baseline on CUB)
R@1 : 67.82%         (paper baseline on CUB)
```

---

## Architecture Summary

| Component | Details |
|---|---|
| Text Encoder | Bi-directional LSTM, 2-layer, hidden=256, embed=300 |
| Image Encoder | Inception-v3 (frozen) + trainable perceptron to D=256 |
| Generator G0 | Noise (z=100) + CA(sentence) → 64×64 |
| Generator G1 | Attention(words, h0) → 128×128 |
| Generator G2 | Attention(words, h1) → 256×256 |
| Discriminators | D0/D1/D2 with Spectral Normalization (our improvement) |

## Loss Function Summary

| Term | Formula | Weight |
|---|---|---|
| GAN unconditional | -½ E[log D(x̂)] | 1 |
| GAN conditional | -½ E[log D(x̂, ē)] | 1 |
| KL divergence | KL(N(μ,σ²)‖N(0,I)) | 1 |
| DAMSM word-level | -Σ log P(D_i‖Q_i) + -Σ log P(Q_i‖D_i) | λ=5 |
| DAMSM sentence-level | (same as above, global features) | λ=5 |

## Key Hyperparameters

| Hyperparameter | Value |
|---|---|
| Noise dimension z | 100 |
| Feature dimension D | 256 |
| Batch size (GAN) | 10 |
| Batch size (DAMSM) | 20 |
| Learning rate | 2e-4 (Adam, β1=0.5) |
| λ (DAMSM weight) | 5 (CUB) |
| γ1, γ2 | 5 |
| γ3 | 10 |
| DAMSM epochs | 120 |
| AttnGAN epochs | 600 |

---

## SOTA Comparison

| Method | CUB IS | COCO IS |
|---|---|---|
| GAN-INT-CLS | 2.88 | 7.88 |
| StackGAN | 3.70 | 8.45 |
| StackGAN++ | 3.82 | — |
| PPGN | — | 9.58 |
| **AttnGAN (paper)** | **4.36** | **25.89** |
| **Ours (AttnGAN+SN)** | *TBD* | — |
