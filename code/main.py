"""
Step 2: Train the full AttnGAN (generators + discriminators).

Requires DAMSM encoders to be pretrained first (pretrain_DAMSM.py).

Usage:
    python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0
    python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0 --resume output/checkpoints/netG_epoch_0100.pth
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets import CUBDataset, collate_fn
from damsm import RNN_ENCODER, CNN_ENCODER
from model import G_NET, D_NET64, D_NET128, D_NET256
from losses import generator_loss, discriminator_loss, kl_loss, damsm_loss
from utils import (load_config, save_models, load_damsm,
                   sample_noise, save_image_grid, Logger)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train AttnGAN')
    p.add_argument('--cfg',    type=str, default='code/cfg/bird_attn2.yml')
    p.add_argument('--gpu',    type=int, default=0)
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--resume', type=str, default='',
                   help='Path to generator checkpoint to resume from')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build discriminators matching the generator stage output sizes
# ---------------------------------------------------------------------------

def build_discriminators(cfg, device):
    df_dim = cfg.DF_DIM
    ef_dim = cfg.EMBEDDING_DIM
    nets_d = [
        D_NET64(df_dim, ef_dim).to(device),
        D_NET128(df_dim, ef_dim).to(device),
        D_NET256(df_dim, ef_dim).to(device),
    ]
    return nets_d


# ---------------------------------------------------------------------------
# One discriminator update step
# ---------------------------------------------------------------------------

def update_discriminators(nets_d, optimizers_d, real_imgs, fake_imgs,
                          sent_emb, wrong_imgs, device, scaler_d):
    """
    Update all discriminators for one batch.
    Each D_i sees real, fake and wrong (real img + mismatched caption).
    """
    d_loss_sum = torch.zeros(1, device=device)
    for d, opt_d, real, fake, wrong in zip(nets_d, optimizers_d,
                                           real_imgs, fake_imgs, wrong_imgs):
        fake = fake.detach()    # do not backprop through G

        with autocast():
            real_uncond, real_cond   = d(real,  sent_emb)
            fake_uncond, fake_cond   = d(fake,  sent_emb)
            wrong_uncond, wrong_cond = d(wrong, sent_emb)
            loss = discriminator_loss(real_uncond, real_cond,
                                      fake_uncond, fake_cond,
                                      wrong_cond)

        opt_d.zero_grad()
        scaler_d.scale(loss).backward()
        scaler_d.step(opt_d)
        d_loss_sum += loss.detach()
    scaler_d.update()
    return d_loss_sum / len(nets_d)    # tensor


# ---------------------------------------------------------------------------
# One generator update step
# ---------------------------------------------------------------------------

def update_generator(g_net, nets_d, optimizer_g,
                     word_embs, sent_emb, local_feat, global_feat,
                     class_ids, z, device, cfg, scaler_g):
    """Update generator using adversarial + DAMSM + KL losses."""
    with autocast():
        fake_imgs, attn_maps, mu, log_var = g_net(z, sent_emb, word_embs)

        g_loss_total = torch.zeros(1, device=device)
        for fake, d in zip(fake_imgs, nets_d):
            uncond, cond = d(fake, sent_emb)
            g_loss_total = g_loss_total + generator_loss(uncond, cond)

        # KL divergence from Conditioning Augmentation
        g_loss_total = g_loss_total + kl_loss(mu, log_var)

        # DAMSM fine-grained matching loss
        # Note: image encoder is frozen during GAN training
        damsm = damsm_loss(word_embs.detach(), sent_emb.detach(),
                           local_feat.detach(), global_feat.detach(),
                           class_ids, z.size(0),
                           cfg.TRAIN.SMOOTH.GAMMA1,
                           cfg.TRAIN.SMOOTH.GAMMA2,
                           cfg.TRAIN.SMOOTH.GAMMA3)
        g_loss_total = g_loss_total + cfg.TRAIN.SMOOTH.LAMBDA * damsm

    optimizer_g.zero_grad()
    scaler_g.scale(g_loss_total).backward()
    scaler_g.step(optimizer_g)
    scaler_g.update()

    return g_loss_total.detach(), fake_imgs, attn_maps


# ---------------------------------------------------------------------------
# Wrong images: shift batch by 1 to get mismatched image-text pairs
# ---------------------------------------------------------------------------

def get_wrong_imgs(imgs_batch, device):
    """Shift each image scale batch by 1 position for mismatched pairs."""
    wrong = []
    for imgs in imgs_batch:
        wrong.append(torch.cat([imgs[1:], imgs[:1]], dim=0).to(device))
    return wrong


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, device, resume_path=''):
    dataset_name = cfg.DATASET_NAME
    data_dir = cfg.DATA_DIR
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoints', dataset_name)
    sample_dir = os.path.join(cfg.OUTPUT_DIR, 'samples', dataset_name)
    log_dir = os.path.join(cfg.LOG_DIR, 'attngan')
    enc_dir = os.path.join(cfg.OUTPUT_DIR, 'DAMSMencoders', dataset_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Dataset ----
    train_set = CUBDataset(data_dir, split='train',
                           words_num=cfg.TEXT.WORDS_NUM,
                           captions_per_image=cfg.TEXT.CAPTIONS_PER_IMAGE)
    test_set  = CUBDataset(data_dir, split='test',
                           words_num=cfg.TEXT.WORDS_NUM,
                           captions_per_image=cfg.TEXT.CAPTIONS_PER_IMAGE)

    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True,
                              collate_fn=collate_fn, drop_last=True)

    n_words = train_set.n_words

    # ---- Load pretrained DAMSM encoders (frozen during GAN training) ----
    text_enc = RNN_ENCODER(n_words,
                           embed_dim=cfg.TEXT.EMBEDDING_DIM,
                           hidden_dim=cfg.EMBEDDING_DIM).to(device)
    img_enc  = CNN_ENCODER(embed_dim=cfg.EMBEDDING_DIM).to(device)
    text_enc, img_enc = load_damsm(text_enc, img_enc, enc_dir, device)
    text_enc.eval()
    img_enc.eval()
    for p in text_enc.parameters():
        p.requires_grad = False
    for p in img_enc.parameters():
        p.requires_grad = False

    # ---- Generator ----
    g_net = G_NET(z_dim=cfg.Z_DIM,
                  ca_dim=cfg.EMBEDDING_DIM // 2,
                  gf_dim=cfg.GF_DIM,
                  ef_dim=cfg.EMBEDDING_DIM,
                  r_num=cfg.R_NUM,
                  num_stages=cfg.TREE.BRANCH_NUM).to(device)

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        g_net.load_state_dict(torch.load(resume_path, map_location=device))
        start_epoch = int(resume_path.split('_epoch_')[1].split('.')[0]) + 1
        print(f'Resumed from epoch {start_epoch - 1}')

    # ---- Discriminators ----
    nets_d = build_discriminators(cfg, device)

    # ---- Optimisers ----
    optimizer_g = optim.Adam(g_net.parameters(),
                             lr=cfg.TRAIN.GENERATOR_LR,
                             betas=(0.5, 0.999))
    optimizers_d = [
        optim.Adam(d.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        for d in nets_d
    ]

    logger = Logger(os.path.join(log_dir, 'train_log.csv'))
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    scaler_g = GradScaler()
    scaler_d = GradScaler()
    global_step = 0

    # Fixed noise for reproducible samples
    fixed_z = sample_noise(cfg.TRAIN.BATCH_SIZE, cfg.Z_DIM, device)

    print(f'Starting AttnGAN training for {cfg.TRAIN.MAX_EPOCH} epochs...')

    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH + 1):
        g_net.train()
        for d in nets_d:
            d.train()

        epoch_g_loss_t = torch.zeros(1, device=device)
        epoch_d_loss_t = torch.zeros(1, device=device)
        t0 = time.time()

        for step, (imgs_batch, word_ids, lengths, class_ids, _) in enumerate(train_loader):
            # ---- Move to device ----
            real_imgs  = [x.to(device, non_blocking=True) for x in imgs_batch]
            word_ids   = word_ids.to(device, non_blocking=True)
            class_ids  = class_ids.to(device, non_blocking=True)

            # ---- Encode text ----
            with torch.no_grad(), autocast():
                word_embs, sent_emb = text_enc(word_ids, lengths)
                # Image features from real high-res images for DAMSM loss
                real_299 = F.interpolate(real_imgs[-1], size=(299, 299),
                                         mode='bilinear', align_corners=False)
                local_feat, global_feat = img_enc(real_299)

            # ---- Sample noise ----
            z = sample_noise(real_imgs[0].size(0), cfg.Z_DIM, device)

            # ---- Generate fakes for discriminator (no G gradient needed here) ----
            with torch.no_grad(), autocast():
                fake_for_d, _, _, _ = g_net(z, sent_emb, word_embs)

            # ---- Wrong images (real + mismatched text) ----
            wrong_imgs = get_wrong_imgs(imgs_batch, device)

            # ---- Update Discriminators ----
            d_loss = update_discriminators(
                nets_d, optimizers_d,
                real_imgs, fake_for_d,
                sent_emb, wrong_imgs, device, scaler_d)

            # ---- Update Generator (generates new fakes internally, with gradients) ----
            g_loss, fake_imgs, attn_maps = update_generator(
                g_net, nets_d, optimizer_g,
                word_embs, sent_emb, local_feat, global_feat,
                class_ids, z, device, cfg, scaler_g)

            epoch_g_loss_t += g_loss
            epoch_d_loss_t += d_loss

            if step % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                writer.add_scalar('AttnGAN/g_loss_step', g_loss.item(), global_step)
                writer.add_scalar('AttnGAN/d_loss_step', d_loss.item(), global_step)
                print(f'[AttnGAN] Epoch {epoch:4d} Step {step:4d} '
                      f'G {g_loss.item():.3f}  D {d_loss.item():.3f}  '
                      f'Time {time.time()-t0:.1f}s')
            global_step += 1

        avg_g = epoch_g_loss_t.item() / len(train_loader)
        avg_d = epoch_d_loss_t.item() / len(train_loader)
        writer.add_scalar('AttnGAN/g_loss_epoch', avg_g, epoch)
        writer.add_scalar('AttnGAN/d_loss_epoch', avg_d, epoch)
        logger.log({'epoch': epoch, 'g_loss': avg_g, 'd_loss': avg_d})

        # Save sample images
        if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch == 1:
            g_net.eval()
            with torch.no_grad():
                sample_imgs, _, _, _ = g_net(fixed_z, sent_emb[:fixed_z.size(0)],
                                             word_embs[:fixed_z.size(0)])
            for scale_idx, imgs in enumerate(sample_imgs):
                scale = [64, 128, 256][scale_idx]
                save_image_grid(
                    imgs,
                    os.path.join(sample_dir, f'epoch_{epoch:04d}_scale{scale}.png')
                )
                grid = (imgs[:8].clamp(-1, 1) + 1) / 2
                writer.add_images(f'Generated/{scale}x{scale}', grid, epoch)

            # Save checkpoint
            save_models(g_net, nets_d, text_enc, img_enc, epoch, ckpt_dir)
            print(f'Saved checkpoint at epoch {epoch}')

    logger.close()
    writer.close()
    print('Training complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.cfg)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print('Warning: running on CPU.')

    print(f'Using device: {device}')
    train(cfg, device, resume_path=args.resume)
