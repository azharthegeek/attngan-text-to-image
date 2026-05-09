# AttnGAN: Fine-Grained Text-to-Image Generation

**Group:** Mohsin Siddiqui (24K-7608) & Muhammad Azhar (24K-7606)  
**Course:** Advanced Computer Vision — NUCES  
**Base paper:** [AttnGAN: Fine-Grained Text to Image Generation with Attentional GANs](https://arxiv.org/pdf/1711.10485) (Xu et al., CVPR 2018)

---

## Overview

This project implements AttnGAN on the CUB-200-2011 birds dataset. Given a free-form text description (e.g., *"A red bird with a short black beak and white belly"*), the model generates a photorealistic bird image through a multi-stage GAN pipeline.

**Our improvements over the paper:**
- **Spectral Normalization** on all discriminator convolutions — enforces a Lipschitz constraint without new hyperparameters
- **Mixed Precision Training (AMP)** — ~40% memory reduction, ~30% faster per step
- **Gradient Checkpointing** — recomputes activations during backward to save ~400 MB on small GPUs
- **Dynamic GPU Tier Detection** — automatically adjusts batch size, model dimensions, and stage count based on available VRAM (4 GB / 8 GB / 16 GB / CPU), so the same code runs on any hardware without manual config edits
- **CUDA allocator tuning** — `expandable_segments:True` on small GPU tiers to reduce OOM from memory fragmentation

---

## Project Structure

```
acv-semester-project/
├── code/
│   ├── cfg/
│   │   └── bird_attn2.yml      # Base hyperparameters (overridden at runtime by GPU tier)
│   ├── datasets.py             # CUB-200-2011 data loader + multi-scale transforms
│   ├── damsm.py                # Text encoder (bi-LSTM) + Image encoder (Inception-v3)
│   ├── model.py                # Generator (G0/G1/G2) + Discriminators with Spectral Norm
│   ├── losses.py               # GAN, KL, and DAMSM losses (5 terms total)
│   ├── utils.py                # GPU tier detection, checkpointing, logging, image saving
│   ├── pretrain_DAMSM.py       # Step 1: pretrain text + image encoders
│   ├── main.py                 # Step 2: train full AttnGAN
│   └── evaluate.py             # Inception Score, R-Precision, single-image inference
├── data/birds/                 # CUB-200-2011 dataset (see Setup)
├── output/
│   ├── DAMSMencoders/birds/    # Pretrained encoder weights
│   ├── checkpoints/birds/      # AttnGAN generator + discriminator checkpoints
│   ├── samples/birds/          # Generated image grids (saved every 50 epochs)
│   └── logs/
│       ├── damsm/              # DAMSM CSV log + TensorBoard events
│       └── attngan/            # AttnGAN CSV log + TensorBoard events
├── generate_report.py          # Generates AttnGAN_Report.docx from training logs
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate virtual environment

```bash
uv venv          # or: python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the CUB-200-2011 dataset

```bash
mkdir -p data/birds

# Download CUB-200-2011 images from:
# http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# Unzip images into data/birds/images/

# Download text annotations + train/test split pickles (Reed et al. format):
# Option A — DF-GAN mirror (no account needed):
pip install gdown
bash scripts/download_birds_text.sh

# Option B — Kaggle (requires free account + kaggle CLI):
# kaggle datasets download -d somthirthabhowmk2001/text-to-image-cub-200-2011
# unzip text-to-image-cub-200-2011.zip -d data/birds/
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
├── train/filenames.pickle
├── test/filenames.pickle
└── class_info.pickle
```

---

## GPU Tier Auto-Configuration

At startup both training scripts detect available VRAM and automatically apply the appropriate settings — **no manual config edits needed**:

| GPU VRAM | Tier | GAN batch | DAMSM batch | Stages | GF / DF | Grad ckpt |
|---|---|---|---|---|---|---|
| ≤ 5 GB (4 GB card) | `4gb` | 2 | 16 | 2 (64+128) | 16 / 32 | Yes |
| ≤ 10 GB | `8gb` | 8 | 32 | 3 (64+128+256) | 32 / 64 | No |
| > 10 GB (Colab) | `16gb` | 16 | 48 | 3 (64+128+256) | 32 / 64 | No |
| CPU | `cpu` | 2 | 8 | 2 (64+128) | 16 / 32 | Yes |

A startup message confirms the detected tier, e.g.:
```
[GPU config] tier=4gb (3.6 GB) | TRAIN batch 10→2, stages=2, GF=16, DF=32, ckpt=True
```

---

## Training

### Step 1 — Pretrain DAMSM encoders

Trains the bi-LSTM text encoder and Inception-v3 image encoder jointly using the DAMSM loss. **Must be run before Step 2.**

```bash
python code/pretrain_DAMSM.py --cfg code/cfg/bird_attn2.yml --gpu 0
```

Encoder weights are saved to `output/DAMSMencoders/birds/` (both per-epoch and `*_latest.pth`).

### Step 2 — Train AttnGAN

Loads the pretrained encoders (frozen) and trains the multi-stage generator and discriminators.

```bash
python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0
```

Resume from a checkpoint:
```bash
python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0 \
    --resume output/checkpoints/birds/netG_epoch_0100.pth
```

Checkpoints are saved every 50 epochs. Sample image grids are written to `output/samples/birds/`.

---

## Monitoring Training

Both scripts write TensorBoard logs to `output/logs/` and a CSV file alongside them.

```bash
tensorboard --logdir output/logs
# Then open http://localhost:6006
```

| Scalar key | Description |
|---|---|
| `DAMSM/loss_step` | DAMSM loss per logged step |
| `DAMSM/loss_epoch` | DAMSM average loss per epoch |
| `DAMSM/lr` | Learning rate schedule |
| `AttnGAN/g_loss_step` | Generator loss per step |
| `AttnGAN/d_loss_step` | Discriminator loss per step |
| `AttnGAN/g_loss_epoch` | Generator average loss per epoch |
| `AttnGAN/d_loss_epoch` | Discriminator average loss per epoch |
| `Generated/64x64`, `128x128` | Sample image grids (logged every 50 epochs) |

On a remote server, forward the port first: `ssh -L 6006:localhost:6006 user@server`

---

## Inference & Evaluation

### Single-image inference (from text description)

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

---

## Architecture Summary

| Component | Details |
|---|---|
| **RNN_ENCODER** | Word embedding (300-d) → bi-LSTM (hidden=256) → word features (B×256×T) + sentence embedding (B×256) |
| **CNN_ENCODER** | Inception-v3 (frozen) → local features (B×256×289 from mixed_6e) + global features (B×256) via learnable projections |
| **CA_NET** | Conditioning Augmentation: ē → μ, log σ → sample c ~ N(μ, σ²) for diversity + KL regularization |
| **G0** | z(100) + c(128) → FC → 4× upsample → 2 ResBlocks → 64×64 image |
| **G1** | hidden + word-attention context → 2× upsample → 2 ResBlocks → 128×128 image |
| **G2** *(8 GB+ only)* | Same as G1, 2× upsample → 256×256 image |
| **D_NET64 / D_NET128** | Spectral-norm convolutions → unconditional head + conditional head (with sentence embedding) |

---

## Loss Function Summary

| Term | Formula | Weight |
|---|---|---|
| `L_D` (per stage) | BCE(real,1) + BCE(fake,0) + BCE(wrong,0) — unconditional + conditional branches | — |
| `L_G` (per stage) | −½ log σ(D_unc(fake)) − ½ log σ(D_cond(fake, text)) | 1.0 |
| `L_KL` | −½ mean(1 + log σ² − μ² − exp(log σ²)) | 1.0 |
| `L_DAMSM` | L₁^w + L₂^w + L₁^s + L₂^s (word & sentence, both directions) | λ = 5 |
| **L_total** | Σ L_{G,i} + L_KL + λ · L_DAMSM | — |

---

## Key Hyperparameters

| Hyperparameter | Value | Note |
|---|---|---|
| Noise dimension z | 100 | Paper default |
| Feature dimension D | 256 | Paper default |
| Word embedding dim | 300 | Paper default |
| Batch size (GAN) | 2 / 8 / 16 | Auto by GPU tier |
| Batch size (DAMSM) | 16 / 32 / 48 | Auto by GPU tier |
| Generator stages | 2 / 3 | Auto by GPU tier |
| Learning rate | 2e-4 | Adam, β1=0.5, β2=0.999 |
| λ (DAMSM weight) | 5 (CUB) | 50 for COCO |
| γ1, γ2 | 5 | Attention normalization |
| γ3 | 10 | DAMSM softmax temperature |
| DAMSM epochs | 120 | ✅ Completed |
| AttnGAN epochs | 600 | Training in progress |

---

## SOTA Comparison (CUB Inception Score)

| Method | CUB IS | COCO IS |
|---|---|---|
| GAN-INT-CLS (2016) | 2.88 | 7.88 |
| StackGAN (2017) | 3.70 | 8.45 |
| StackGAN++ (2018) | 3.82 | 9.58 |
| **AttnGAN (paper, 2018)** | **4.36** | **25.89** |
| **Ours (AttnGAN + improvements)** | *Training in progress* | — |
