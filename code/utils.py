"""Utility functions: checkpointing, logging, image grids, config loading."""

import os
import yaml
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(cfg_path):
    """Load YAML config file and return an EasyDict."""
    with open(cfg_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    return cfg


def get_gpu_tier(device):
    """Return tier label based on total VRAM: '4gb', '8gb', '16gb', or 'cpu'."""
    if device.type != 'cuda':
        return 'cpu'
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if total_gb <= 5:
        return '4gb'
    elif total_gb <= 10:
        return '8gb'
    return '16gb'


def apply_gpu_memory_config(cfg, device):
    """Override cfg in-place with tier-appropriate batch sizes, model dims, and worker counts."""
    tier = get_gpu_tier(device)
    tier_settings = {
        'cpu':  dict(batch=2, damsm_batch=8,  workers=0, gf=16, df=32, stages=2, ckpt=True),
        '4gb':  dict(batch=2, damsm_batch=16, workers=2, gf=16, df=32, stages=2, ckpt=True),
        '8gb':  dict(batch=8, damsm_batch=32, workers=4, gf=32, df=64, stages=3, ckpt=False),
        '16gb': dict(batch=16,damsm_batch=48, workers=8, gf=32, df=64, stages=3, ckpt=False),
    }
    s = tier_settings[tier]

    orig_batch = cfg.TRAIN.BATCH_SIZE
    cfg.TRAIN.BATCH_SIZE    = s['batch']
    cfg.TRAIN.NUM_WORKERS   = s['workers']
    cfg.DAMSM.BATCH_SIZE    = s['damsm_batch']
    cfg.DAMSM.NUM_WORKERS   = s['workers']
    cfg.GF_DIM              = s['gf']
    cfg.DF_DIM              = s['df']
    cfg.TREE.BRANCH_NUM     = s['stages']
    cfg.USE_GRAD_CHECKPOINT = s['ckpt']

    # Enable expandable allocator on small GPUs to reduce fragmentation OOMs
    if tier in ('4gb', 'cpu'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    total_label = (
        f"{torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB"
        if device.type == 'cuda' else 'CPU'
    )
    print(f"[GPU config] tier={tier} ({total_label}) | "
          f"TRAIN batch {orig_batch}→{s['batch']}, stages={s['stages']}, "
          f"GF={s['gf']}, DF={s['df']}, ckpt={s['ckpt']}, workers={s['workers']}")
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_models(g_net, discriminators, text_enc, img_enc, epoch, out_dir):
    """Save generator, discriminators and encoders to output directory."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(g_net.state_dict(),
               os.path.join(out_dir, f'netG_epoch_{epoch:04d}.pth'))
    for i, d in enumerate(discriminators):
        torch.save(d.state_dict(),
                   os.path.join(out_dir, f'netD{i}_epoch_{epoch:04d}.pth'))
    torch.save(text_enc.state_dict(),
               os.path.join(out_dir, f'text_encoder_{epoch:04d}.pth'))
    torch.save(img_enc.state_dict(),
               os.path.join(out_dir, f'image_encoder_{epoch:04d}.pth'))


def save_damsm(text_enc, img_enc, epoch, out_dir):
    """Save DAMSM encoder weights during pretraining."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(text_enc.state_dict(),
               os.path.join(out_dir, f'text_encoder_{epoch:04d}.pth'))
    torch.save(img_enc.state_dict(),
               os.path.join(out_dir, f'image_encoder_{epoch:04d}.pth'))
    # Also keep a 'latest' copy for easy loading
    torch.save(text_enc.state_dict(),
               os.path.join(out_dir, 'text_encoder_latest.pth'))
    torch.save(img_enc.state_dict(),
               os.path.join(out_dir, 'image_encoder_latest.pth'))


def load_damsm(text_enc, img_enc, enc_dir, device):
    """Load latest DAMSM encoder weights."""
    t_path = os.path.join(enc_dir, 'text_encoder_latest.pth')
    i_path = os.path.join(enc_dir, 'image_encoder_latest.pth')
    text_enc.load_state_dict(torch.load(t_path, map_location=device))
    img_enc.load_state_dict(torch.load(i_path, map_location=device))
    print(f'Loaded DAMSM encoders from {enc_dir}')
    return text_enc, img_enc


# ---------------------------------------------------------------------------
# Image grid saving
# ---------------------------------------------------------------------------

def tensor_to_pil(t):
    """Convert normalised [-1, 1] tensor (C, H, W) to PIL image."""
    img = t.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0          # [-1,1] → [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return Image.fromarray(img)


def save_image_grid(imgs_batch, save_path, nrow=8):
    """
    Save a batch of images as a grid.

    Args:
        imgs_batch : (batch, 3, H, W) tensor in [-1, 1]
        save_path  : output file path (.png or .jpg)
        nrow       : images per row in the grid
    """
    batch = imgs_batch.size(0)
    H = imgs_batch.size(2)
    W = imgs_batch.size(3)
    ncol = (batch + nrow - 1) // nrow
    grid = Image.new('RGB', (nrow * W, ncol * H))

    for idx in range(batch):
        img = tensor_to_pil(imgs_batch[idx])
        row = idx // nrow
        col = idx % nrow
        grid.paste(img, (col * W, row * H))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid.save(save_path)


# ---------------------------------------------------------------------------
# Noise vector sampling
# ---------------------------------------------------------------------------

def sample_noise(batch_size, z_dim, device):
    """Sample z ~ N(0, I)."""
    return torch.randn(batch_size, z_dim, device=device)


# ---------------------------------------------------------------------------
# Attention map visualisation
# ---------------------------------------------------------------------------

def save_attention_map(attn, words, img, save_path, top_k=5):
    """
    Overlay the top-k attended words on the image and save.

    Args:
        attn      : (T, N) numpy array — attention weights
        words     : list of word strings (length T)
        img       : PIL image
        save_path : output file path
        top_k     : number of top attended words to highlight
    """
    # Sum attention over spatial regions to get per-word importance
    word_importance = attn.sum(axis=1)   # (T,)
    top_idx = word_importance.argsort()[::-1][:top_k]
    top_words = [(words[i] if i < len(words) else '') for i in top_idx]

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    text = ' | '.join(top_words)
    draw.text((5, 5), text, fill=(255, 0, 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Simple CSV logger for training metrics."""

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.path = log_path
        self.file = open(log_path, 'w')
        self._header_written = False

    def log(self, metrics: dict):
        if not self._header_written:
            self.file.write(','.join(metrics.keys()) + '\n')
            self._header_written = True
        self.file.write(','.join(str(v) for v in metrics.values()) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()
