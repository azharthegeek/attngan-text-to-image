"""
Step 1: Pretrain the DAMSM (text encoder + image encoder).

Run BEFORE training the full AttnGAN.
Saves encoder weights to output/DAMSMencoders/<dataset_name>/

Usage:
    python code/pretrain_DAMSM.py --cfg code/cfg/bird_attn2.yml --gpu 0
"""

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make sure the code/ directory is on the path when calling from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets import CUBDataset, collate_fn
from damsm import RNN_ENCODER, CNN_ENCODER
from losses import damsm_loss
from utils import load_config, save_damsm, Logger, apply_gpu_memory_config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Pretrain DAMSM encoders')
    p.add_argument('--cfg',  type=str, default='code/cfg/bird_attn2.yml',
                   help='Path to YAML config file')
    p.add_argument('--gpu',  type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset + loader for DAMSM  (images at 299×299 for Inception-v3)
# ---------------------------------------------------------------------------

class DAMSMTransform:
    """Resize to 299×299 as required by Inception-v3."""

    def __init__(self, split='train'):
        if split == 'train':
            self.t = T.Compose([
                T.Resize(int(299 * 76 / 64)),
                T.RandomCrop(299),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.t = T.Compose([
                T.Resize(int(299 * 76 / 64)),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __call__(self, img):
        return self.t(img)


class DAMSMDataset(CUBDataset):
    """CUBDataset variant that returns 299×299 images for DAMSM training."""

    def __init__(self, *args, **kwargs):
        split = kwargs.get('split', 'train')
        super().__init__(*args, **kwargs)
        self._damsm_transform = DAMSMTransform(split)

    def _load_image(self, fname):
        from PIL import Image as PILImage
        img_path = os.path.join(self.data_dir, 'images', fname + '.jpg')
        img = PILImage.open(img_path).convert('RGB')
        return self._damsm_transform(img)   # returns single tensor (3, 299, 299)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        class_id = self.class_info[idx]
        img = self._load_image(fname)

        import numpy as np
        if self.split == 'train':
            cap_idx = np.random.randint(0, len(self.captions[idx]))
        else:
            cap_idx = 0
        caption = self.captions[idx][cap_idx]
        word_ids, length = self._caption_to_ids(caption)

        import torch
        word_ids = torch.LongTensor(word_ids)
        return img, word_ids, length, class_id, fname


def damsm_collate(batch):
    imgs, word_ids, lengths, class_ids, fnames = zip(*batch)
    imgs = torch.stack(imgs)
    word_ids = torch.stack(word_ids)
    lengths = torch.LongTensor(lengths)
    class_ids = torch.LongTensor(class_ids)
    return imgs, word_ids, lengths, class_ids, fnames


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_damsm(cfg, device):
    dataset_name = cfg.DATASET_NAME
    data_dir = cfg.DATA_DIR
    out_dir = os.path.join(cfg.OUTPUT_DIR, 'DAMSMencoders', dataset_name)
    log_dir = os.path.join(cfg.LOG_DIR, 'damsm')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Dataset ----
    train_set = DAMSMDataset(data_dir, split='train',
                             words_num=cfg.TEXT.WORDS_NUM,
                             captions_per_image=cfg.TEXT.CAPTIONS_PER_IMAGE)
    test_set  = DAMSMDataset(data_dir, split='test',
                             words_num=cfg.TEXT.WORDS_NUM,
                             captions_per_image=cfg.TEXT.CAPTIONS_PER_IMAGE)

    nw = cfg.DAMSM.NUM_WORKERS
    train_loader = DataLoader(train_set, batch_size=cfg.DAMSM.BATCH_SIZE,
                              shuffle=True,  num_workers=nw,
                              pin_memory=(nw > 0), persistent_workers=(nw > 0),
                              collate_fn=damsm_collate, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg.DAMSM.BATCH_SIZE,
                              shuffle=False, num_workers=nw,
                              pin_memory=(nw > 0), persistent_workers=(nw > 0),
                              collate_fn=damsm_collate, drop_last=True)

    n_words = train_set.n_words
    print(f'Vocabulary size: {n_words}')

    # ---- Models ----
    text_enc = RNN_ENCODER(n_words,
                           embed_dim=cfg.TEXT.EMBEDDING_DIM,
                           hidden_dim=cfg.EMBEDDING_DIM).to(device)
    img_enc  = CNN_ENCODER(embed_dim=cfg.EMBEDDING_DIM).to(device)

    # Only train the newly added projection layers in CNN_ENCODER + the full RNN
    params = list(text_enc.parameters()) + \
             list(img_enc.local_proj.parameters()) + \
             list(img_enc.global_proj.parameters())
    optimizer = optim.Adam(params, lr=cfg.DAMSM.LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    logger = Logger(os.path.join(log_dir, 'train_log.csv'))
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    best_loss = float('inf')
    scaler = GradScaler()
    global_step = 0

    for epoch in range(1, cfg.DAMSM.MAX_EPOCH + 1):
        text_enc.train()
        img_enc.train()
        epoch_loss_t = torch.zeros(1, device=device)
        t0 = time.time()

        for step, (imgs, word_ids, lengths, class_ids, _) in enumerate(train_loader):
            imgs      = imgs.to(device, non_blocking=True)
            word_ids  = word_ids.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)

            with autocast():
                # Forward text encoder
                word_embs, sent_emb = text_enc(word_ids, lengths)

                # Forward image encoder (resize already done in dataset)
                local_feat, global_feat = img_enc(imgs)

                # DAMSM loss
                loss = damsm_loss(word_embs, sent_emb,
                                  local_feat, global_feat,
                                  class_ids, imgs.size(0),
                                  cfg.TRAIN.SMOOTH.GAMMA1,
                                  cfg.TRAIN.SMOOTH.GAMMA2,
                                  cfg.TRAIN.SMOOTH.GAMMA3)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Clip gradients for stable RNN training
            torch.nn.utils.clip_grad_norm_(text_enc.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_t += loss.detach()

            if step % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                writer.add_scalar('DAMSM/loss_step', loss.item(), global_step)
                print(f'[DAMSM] Epoch {epoch:3d} Step {step:4d} '
                      f'Loss {loss.item():.4f}  '
                      f'Time {time.time()-t0:.1f}s')
            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss_t.item() / len(train_loader)
        writer.add_scalar('DAMSM/loss_epoch', avg_loss, epoch)
        writer.add_scalar('DAMSM/lr', scheduler.get_last_lr()[0], epoch)
        logger.log({'epoch': epoch, 'train_loss': avg_loss})
        print(f'[DAMSM] Epoch {epoch} done | avg loss: {avg_loss:.4f}')

        # Save best encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_damsm(text_enc, img_enc, epoch, out_dir)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_damsm(text_enc, img_enc, epoch, out_dir)

    logger.close()
    writer.close()
    print(f'DAMSM pretraining done. Encoders saved to {out_dir}')


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
        print('Warning: running on CPU — this will be slow.')

    print(f'Using device: {device}')
    cfg = apply_gpu_memory_config(cfg, device)
    train_damsm(cfg, device)
