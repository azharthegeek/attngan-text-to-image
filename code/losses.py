"""
All loss functions for AttnGAN training.

Five loss terms (Section 3 of the paper):
  1. L_G_uncond   — GAN unconditional generator loss
  2. L_G_cond     — GAN conditional generator loss  (image + text matching)
  3. L_D           — Discriminator adversarial loss (both branches)
  4. L_DAMSM      — word-level + sentence-level matching loss (eq. 12-14)
  5. L_KL          — KL divergence from Conditioning Augmentation

Total generator objective (eq. 3):
    L = L_G + λ * L_DAMSM     (λ = 5 for CUB, 50 for COCO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GAN losses
# ---------------------------------------------------------------------------

def generator_loss(uncond_pred, cond_pred):
    """
    Non-saturating GAN loss for one generator stage.
    The generator wants D(x̂) → 1 and D(x̂, ē) → 1.

    Args:
        uncond_pred : (batch,)  discriminator unconditional output for fake
        cond_pred   : (batch,)  discriminator conditional output for fake
    Returns:
        scalar loss
    """
    loss_uncond = -0.5 * torch.mean(torch.log(torch.sigmoid(uncond_pred) + 1e-8))
    loss_cond   = -0.5 * torch.mean(torch.log(torch.sigmoid(cond_pred)   + 1e-8))
    return loss_uncond + loss_cond


def discriminator_loss(real_uncond, real_cond, fake_uncond, fake_cond, wrong_cond):
    """
    Cross-entropy adversarial loss for discriminator (eq. 4-5 of paper).

    Uses 3 inputs per batch:
      - real image + matching text    → real_cond should be 1
      - real image + mismatched text  → wrong_cond should be 0
      - fake image + matching text    → fake_cond should be 0
      - real image (unconditional)    → real_uncond should be 1
      - fake image (unconditional)    → fake_uncond should be 0

    Args:
        real_uncond  : (batch,)
        real_cond    : (batch,)
        fake_uncond  : (batch,)
        fake_cond    : (batch,)
        wrong_cond   : (batch,) — real image with wrong caption
    Returns:
        scalar loss
    """
    # Unconditional branch
    real_labels  = torch.ones_like(real_uncond)
    fake_labels  = torch.zeros_like(fake_uncond)

    loss_uncond = 0.5 * (
        F.binary_cross_entropy_with_logits(real_uncond, real_labels) +
        F.binary_cross_entropy_with_logits(fake_uncond, fake_labels)
    )

    # Conditional branch
    loss_cond = 0.5 * (
        F.binary_cross_entropy_with_logits(real_cond,  real_labels) +
        0.5 * F.binary_cross_entropy_with_logits(fake_cond,  fake_labels) +
        0.5 * F.binary_cross_entropy_with_logits(wrong_cond, fake_labels)
    )

    return loss_uncond + loss_cond


# ---------------------------------------------------------------------------
# KL Divergence — Conditioning Augmentation
# ---------------------------------------------------------------------------

def kl_loss(mu, log_var):
    """
    KL( N(mu, sigma^2) || N(0, I) )
    = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ---------------------------------------------------------------------------
# DAMSM Loss  (Section 3.2, equations 11-14)
# ---------------------------------------------------------------------------

def damsm_loss(word_embs, sent_emb, local_feat, global_feat,
               class_ids, batch_size, gamma1, gamma2, gamma3):
    """
    Deep Attentional Multimodal Similarity Model loss.

    Computes four terms:
      L1^w — word-level image-to-text loss
      L2^w — word-level text-to-image loss
      L1^s — sentence-level image-to-text loss
      L2^s — sentence-level text-to-image loss

    Args:
        word_embs   : (batch, D, T)   — word feature matrix e from text encoder
        sent_emb    : (batch, D)      — sentence feature ē
        local_feat  : (batch, D, 289) — local image features v
        global_feat : (batch, D)      — global image feature v̄
        class_ids   : (batch,)        — bird class IDs (used to mask same-class pairs)
        batch_size  : int
        gamma1,2,3  : hyperparameters (γ1=5, γ2=5, γ3=10 from paper)
    Returns:
        total DAMSM loss (scalar)
    """
    # --- Word-level matching score R(Q_i, D_j)  (equations 7-10) ---
    # For each (image Q_i, description D_j) pair compute R

    # We compute the full batch×batch score matrix
    w_scores = _word_level_scores(word_embs, local_feat, gamma1, gamma2)
    # w_scores: (batch, batch)  entry [i,j] = R(image_i, text_j)

    # --- Sentence-level matching score (simple cosine) ---
    s_scores = _sent_level_scores(global_feat, sent_emb)
    # s_scores: (batch, batch)  entry [i,j] = cos(global_i, sent_j)

    # Mask out pairs from the same class (they can legitimately match)
    mask = _same_class_mask(class_ids, batch_size, word_embs.device)

    w_loss1, w_loss2 = _matching_losses(w_scores, mask, gamma3, batch_size)
    s_loss1, s_loss2 = _matching_losses(s_scores, mask, gamma3, batch_size)

    return w_loss1 + w_loss2 + s_loss1 + s_loss2


# -- helpers ----------------------------------------------------------------

def _word_level_scores(word_embs, local_feat, gamma1, gamma2):
    """
    Compute R(Q_i, D_j) for all (i,j) pairs.
    Returns (batch, batch) score matrix where [i,j] = R(image_i, text_j).
    """
    batch = word_embs.size(0)
    T = word_embs.size(2)
    D = word_embs.size(1)

    scores = torch.zeros(batch, batch, device=word_embs.device)

    for i in range(batch):
        # word features for text i: (D, T) → (T, D)
        e = word_embs[i].t()                              # (T, D)
        e = F.normalize(e, p=2, dim=1)

        for j in range(batch):
            # local features for image j: (D, N)
            v = local_feat[j]                             # (D, N)
            v = F.normalize(v, p=2, dim=0)

            # Similarity matrix s = e^T * v: (T, N)
            s = torch.mm(e, v)                            # (T, N)

            # Normalise over N (eq. 8)
            s_bar = F.softmax(s, dim=1)                   # (T, N)

            # Attention weights alpha: softmax(gamma1 * s_bar) over N
            alpha = F.softmax(gamma1 * s_bar, dim=1)      # (T, N)

            # Region context c_i = sum_j alpha_j * v_j: (T, D)
            c = torch.mm(alpha, v.t())                    # (T, D)
            c = F.normalize(c, p=2, dim=1)

            # Cosine similarity R(c_i, e_i) per word: (T,)
            r_words = (c * e).sum(dim=1)                  # (T,)

            # Aggregate via logsumexp (eq. 10)
            r = torch.log(torch.clamp(
                torch.sum(torch.exp(gamma2 * r_words)), min=1e-8
            )) / gamma2

            scores[i, j] = r

    return scores


def _sent_level_scores(global_feat, sent_emb):
    """Compute cosine similarity matrix between all global image/text pairs."""
    g = F.normalize(global_feat, p=2, dim=1)   # (batch, D)
    s = F.normalize(sent_emb,   p=2, dim=1)   # (batch, D)
    return torch.mm(g, s.t())                  # (batch, batch)


def _same_class_mask(class_ids, batch_size, device):
    """Create boolean mask: True where images belong to the same class."""
    mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j and class_ids[i] == class_ids[j]:
                mask[i, j] = True
    return mask


def _matching_losses(scores, mask, gamma3, batch_size):
    """
    Compute bidirectional matching losses (equations 11-13 of the paper).

    P(D_i | Q_i) = softmax(gamma3 * R(Q_i, D)) over descriptions D
    L1 = -sum log P(D_i | Q_i)   [image → text]
    L2 = -sum log P(Q_i | D_i)   [text → image]

    Same-class pairs are masked to 0 before softmax so they don't count
    as negatives (same bird species can have similar descriptions).
    """
    # Mask same-class off-diagonal entries
    scores_masked = scores.clone()
    scores_masked[mask] = -float('inf')

    # L1: image→text  (row = image, column = text)
    log_probs_l1 = F.log_softmax(gamma3 * scores_masked, dim=1)
    l1 = -torch.mean(torch.diag(log_probs_l1))

    # L2: text→image  (column = image, row = text)
    log_probs_l2 = F.log_softmax(gamma3 * scores_masked.t(), dim=1)
    l2 = -torch.mean(torch.diag(log_probs_l2))

    return l1, l2
