"""
Evaluation script: Inception Score (IS) and R-Precision.

Usage — generate samples and evaluate:
    python code/evaluate.py --cfg code/cfg/bird_attn2.yml \
           --model output/checkpoints/birds/netG_epoch_0600.pth \
           --gpu 0

Usage — text-to-image inference (single caption):
    python code/evaluate.py --cfg code/cfg/bird_attn2.yml \
           --model output/checkpoints/birds/netG_epoch_0600.pth \
           --text "this bird is red with white belly and short beak" \
           --gpu 0 --out samples/custom_bird.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from PIL import Image
from scipy.stats import entropy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets import CUBDataset, collate_fn
from damsm import RNN_ENCODER, CNN_ENCODER
from model import G_NET
from utils import load_config, load_damsm, sample_noise, tensor_to_pil


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate AttnGAN')
    p.add_argument('--cfg',   type=str, default='code/cfg/bird_attn2.yml')
    p.add_argument('--model', type=str, required=True,
                   help='Path to trained generator .pth file')
    p.add_argument('--gpu',   type=int, default=0)
    p.add_argument('--text',  type=str, default='',
                   help='Single caption for inference mode')
    p.add_argument('--out',   type=str, default='',
                   help='Output path for single-image inference')
    p.add_argument('--n_samples', type=int, default=30000,
                   help='Number of images for IS computation')
    p.add_argument('--splits', type=int, default=10,
                   help='Number of splits for IS')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

class InceptionV3Features(torch.nn.Module):
    """Wrapper around Inception-v3 to extract softmax probabilities."""

    def __init__(self):
        super().__init__()
        inception = tv_models.inception_v3(pretrained=True, aux_logits=False)
        inception.eval()
        self.inception = inception

    @torch.no_grad()
    def forward(self, x):
        # x: (batch, 3, H, W) in [-1, 1]
        # Inception expects 299×299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Rescale from [-1,1] → [0,1] then apply Inception normalisation
        x = (x + 1) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        logits = self.inception(x)
        probs = F.softmax(logits, dim=1)
        return probs


def compute_inception_score(probs_all, splits=10):
    """
    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )
    Splits are used to estimate mean and std.
    """
    scores = []
    n = probs_all.shape[0]
    split_size = n // splits
    for k in range(splits):
        part = probs_all[k * split_size: (k + 1) * split_size]
        py = part.mean(axis=0)          # marginal distribution
        kl = [entropy(pyx, py) for pyx in part]
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# R-Precision
# ---------------------------------------------------------------------------

def compute_r_precision(g_net, text_enc, img_enc, test_loader,
                        cfg, device, R=1, n_batches=100):
    """
    R-Precision: generate images from test captions, then retrieve the correct
    caption from a pool of 100 (1 ground truth + 99 mismatches).

    Uses the pretrained DAMSM encoders to embed both generated images and captions.
    Returns r-precision@R.
    """
    g_net.eval()
    text_enc.eval()
    img_enc.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (imgs_batch, word_ids, lengths, class_ids, _) in enumerate(test_loader):
            if batch_idx >= n_batches:
                break

            word_ids = word_ids.to(device)
            class_ids = class_ids.to(device)
            batch_size = word_ids.size(0)

            word_embs, sent_emb = text_enc(word_ids, lengths)

            z = sample_noise(batch_size, cfg.Z_DIM, device)
            fake_imgs, _, _, _ = g_net(z, sent_emb, word_embs)

            # Get image embeddings for the generated 256×256 images
            fake_256 = fake_imgs[-1]
            fake_299 = F.interpolate(fake_256, size=(299, 299),
                                     mode='bilinear', align_corners=False)
            _, fake_global = img_enc(fake_299)   # (batch, D)

            # Normalise
            fake_global = F.normalize(fake_global, p=2, dim=1)
            sent_emb_n  = F.normalize(sent_emb,   p=2, dim=1)

            # Cosine similarity: each generated image vs all captions in batch
            sim = torch.mm(fake_global, sent_emb_n.t())   # (batch, batch)

            # For each image, check if the matching caption is in top-R
            for i in range(batch_size):
                top_r = sim[i].topk(R).indices.tolist()
                if i in top_r:
                    correct += 1
                total += 1

    r_precision = correct / total if total > 0 else 0.0
    return r_precision


# ---------------------------------------------------------------------------
# Single caption inference
# ---------------------------------------------------------------------------

def infer_from_text(g_net, text_enc, word2idx, caption, cfg, device, out_path):
    """Generate an image from a single text caption and save it."""
    import nltk
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    word_ids = [word2idx.get(tok, 0) for tok in tokens[:cfg.TEXT.WORDS_NUM]]
    length = len(word_ids)
    while len(word_ids) < cfg.TEXT.WORDS_NUM:
        word_ids.append(0)

    word_ids_t = torch.LongTensor(word_ids).unsqueeze(0).to(device)
    length_t   = torch.LongTensor([length])

    g_net.eval()
    text_enc.eval()
    with torch.no_grad():
        word_embs, sent_emb = text_enc(word_ids_t, length_t)
        z = sample_noise(1, cfg.Z_DIM, device)
        fake_imgs, _, _, _ = g_net(z, sent_emb, word_embs)

    # Save all 3 scales side by side
    pils = [tensor_to_pil(fake_imgs[i][0]) for i in range(len(fake_imgs))]
    widths = [p.width for p in pils]
    heights = [p.height for p in pils]
    combined = Image.new('RGB', (sum(widths), max(heights)))
    x_offset = 0
    for p in pils:
        combined.paste(p, (x_offset, 0))
        x_offset += p.width

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    combined.save(out_path)
    print(f'Saved generated image to {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    # ---- Load dataset (for vocab and test loader) ----
    test_set = CUBDataset(cfg.DATA_DIR, split='test',
                          words_num=cfg.TEXT.WORDS_NUM,
                          captions_per_image=cfg.TEXT.CAPTIONS_PER_IMAGE)
    test_loader = DataLoader(test_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                             shuffle=False, num_workers=4,
                             collate_fn=collate_fn, drop_last=True)
    n_words = test_set.n_words

    # ---- Load encoders ----
    text_enc = RNN_ENCODER(n_words,
                           embed_dim=cfg.TEXT.EMBEDDING_DIM,
                           hidden_dim=cfg.EMBEDDING_DIM).to(device)
    img_enc  = CNN_ENCODER(embed_dim=cfg.EMBEDDING_DIM).to(device)
    enc_dir = os.path.join(cfg.OUTPUT_DIR, 'DAMSMencoders', cfg.DATASET_NAME)
    text_enc, img_enc = load_damsm(text_enc, img_enc, enc_dir, device)

    # ---- Load generator ----
    g_net = G_NET(z_dim=cfg.Z_DIM,
                  ca_dim=cfg.EMBEDDING_DIM // 2,
                  gf_dim=cfg.GF_DIM,
                  ef_dim=cfg.EMBEDDING_DIM,
                  r_num=cfg.R_NUM,
                  num_stages=cfg.TREE.BRANCH_NUM).to(device)
    g_net.load_state_dict(torch.load(args.model, map_location=device))
    g_net.eval()

    # ---- Inference mode ----
    if args.text:
        out = args.out if args.out else 'output/inference.png'
        infer_from_text(g_net, text_enc, test_set.word2idx,
                        args.text, cfg, device, out)
        return

    # ---- Compute Inception Score ----
    print(f'Generating {args.n_samples} images for IS computation...')
    inception = InceptionV3Features().to(device)

    all_probs = []
    generated = 0
    with torch.no_grad():
        for imgs_batch, word_ids, lengths, class_ids, _ in test_loader:
            if generated >= args.n_samples:
                break
            word_ids = word_ids.to(device)
            word_embs, sent_emb = text_enc(word_ids, lengths)
            z = sample_noise(word_ids.size(0), cfg.Z_DIM, device)
            fake_imgs, _, _, _ = g_net(z, sent_emb, word_embs)

            probs = inception(fake_imgs[-1]).cpu().numpy()
            all_probs.append(probs)
            generated += probs.shape[0]
            print(f'  Generated {generated}/{args.n_samples}', end='\r')

    all_probs = np.concatenate(all_probs, axis=0)[:args.n_samples]
    is_mean, is_std = compute_inception_score(all_probs, splits=args.splits)
    print(f'\nInception Score: {is_mean:.2f} ± {is_std:.2f}')

    # ---- Compute R-Precision ----
    print('Computing R-Precision...')
    r_prec = compute_r_precision(g_net, text_enc, img_enc, test_loader,
                                 cfg, device, R=1)
    print(f'R-Precision@1: {r_prec * 100:.2f}%')

    print(f'\n=== Results ===')
    print(f'IS  : {is_mean:.2f} ± {is_std:.2f}  (paper: 4.36 ± 0.03 on CUB)')
    print(f'R@1 : {r_prec * 100:.2f}%          (paper: 67.82 ± 4.43% on CUB)')


if __name__ == '__main__':
    main()
