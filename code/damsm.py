"""
Deep Attentional Multimodal Similarity Model (DAMSM).

Contains:
  - RNN_ENCODER  : bi-directional LSTM text encoder
  - CNN_ENCODER  : Inception-v3 image encoder (mixed_6e + avg-pool)
  - cosine_similarity_matrix : utility used by the DAMSM loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Text Encoder — Bi-directional LSTM
# ---------------------------------------------------------------------------

class RNN_ENCODER(nn.Module):
    """
    Encodes a variable-length word sequence into:
      - word features  e  : (batch, D, T)  — one D-dim vector per word
      - sentence feature ē : (batch, D)    — last hidden state (fwd+bwd)

    Architecture (Section 3.2 of the paper):
      embedding(vocab_size, embed_dim) → bi-LSTM(embed_dim, D//2, num_layers)
    The two directions are concatenated so the hidden state is D-dimensional.
    """

    def __init__(self, n_words, embed_dim=300, hidden_dim=256, num_layers=1,
                 dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(n_words, embed_dim)
        self.drop = nn.Dropout(dropout)
        # bidirectional LSTM; each direction has hidden_dim//2 so output = hidden_dim
        self.rnn = nn.LSTM(embed_dim, hidden_dim // 2, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, captions, cap_lens, hidden=None):
        """
        Args:
            captions : LongTensor (batch, T)
            cap_lens : LongTensor (batch,) — actual sequence lengths
        Returns:
            word_embs  : (batch, D, T)
            sent_emb   : (batch, D)
        """
        emb = self.drop(self.embedding(captions))   # (batch, T, embed_dim)

        # pack for variable lengths
        cap_lens_cpu = cap_lens.cpu().tolist()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, cap_lens_cpu, batch_first=True, enforce_sorted=False)
        output, (h_n, _) = self.rnn(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=captions.size(1))

        # word features: (batch, T, D) → (batch, D, T)
        word_embs = output.transpose(1, 2)

        # sentence feature: concatenate last hidden states of fwd and bwd
        # h_n shape: (num_layers*2, batch, D//2)
        h_n = h_n.view(self.num_layers, 2, captions.size(0), self.hidden_dim // 2)
        h_fwd = h_n[-1, 0]   # (batch, D//2)
        h_bwd = h_n[-1, 1]   # (batch, D//2)
        sent_emb = torch.cat([h_fwd, h_bwd], dim=1)   # (batch, D)

        return word_embs, sent_emb


# ---------------------------------------------------------------------------
# Image Encoder — Inception-v3
# ---------------------------------------------------------------------------

class CNN_ENCODER(nn.Module):
    """
    Extracts local and global image features using Inception-v3.

    Local features  v : (batch, D, 289)   from mixed_6e (17×17 = 289 sub-regions)
    Global features v̄ : (batch, D)         from last avg-pool layer

    A single trainable perceptron maps both to the D-dim multimodal space.
    All Inception-v3 weights are frozen; only the perceptron is trained.
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Load Inception-v3; disable aux classifier
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.aux_logits = False

        # Split at mixed_6e to get local features
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e   # local feature output (768 ch, 17×17)

        # Layers after mixed_6e for global features
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        # Trainable projection: 768 → D (local), 2048 → D (global)
        self.local_proj = nn.Linear(768, embed_dim)
        self.global_proj = nn.Linear(2048, embed_dim)

        # Freeze all Inception-v3 parameters
        for name, param in self.named_parameters():
            if 'local_proj' not in name and 'global_proj' not in name:
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x : (batch, 3, 299, 299)  — normalised image
        Returns:
            local_feat  : (batch, D, 289)
            global_feat : (batch, D)
        """
        # Inception forward up to mixed_6e
        x = self.Conv2d_1a_3x3(x)    # 149×149
        x = self.Conv2d_2a_3x3(x)    # 147×147
        x = self.Conv2d_2b_3x3(x)    # 147×147
        x = F.max_pool2d(x, 3, stride=2)  # 73×73
        x = self.Conv2d_3b_1x1(x)    # 73×73
        x = self.Conv2d_4a_3x3(x)    # 71×71
        x = F.max_pool2d(x, 3, stride=2)  # 35×35
        x = self.Mixed_5b(x)          # 35×35
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)          # 17×17
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)          # (batch, 768, 17, 17)

        # local features: flatten spatial dims → project
        local_raw = x.view(x.size(0), 768, -1)           # (batch, 768, 289)
        local_feat = self.local_proj(local_raw.permute(0, 2, 1))  # (batch, 289, D)
        local_feat = local_feat.permute(0, 2, 1)          # (batch, D, 289)

        # global features: continue through inception, avg-pool
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))              # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)                         # (batch, 2048)
        global_feat = self.global_proj(x)                  # (batch, D)

        return local_feat, global_feat


# ---------------------------------------------------------------------------
# Utility — normalised cosine similarity used inside DAMSM loss
# ---------------------------------------------------------------------------

def l2_norm(x):
    """L2-normalise along last dimension."""
    return F.normalize(x, p=2, dim=-1)


def cosine_similarity_matrix(word_embs, local_feat, gamma1):
    """
    Compute attention-driven image-text matching score R(Q, D) for a batch.

    Paper Section 3.2 — Equations 7-10.

    Args:
        word_embs  : (batch, D, T)   — word feature matrix e
        local_feat : (batch, D, N)   — local image features v  (N=289)
        gamma1     : float           — attention normalisation factor
    Returns:
        scores : (batch, batch)  — R(Q_i, D_j) for all pairs
    """
    batch = word_embs.size(0)
    T = word_embs.size(2)
    N = local_feat.size(2)
    D = word_embs.size(1)

    # Expand and compute raw similarities: s = e^T v  (eq. 7)
    # word_embs: (batch, D, T)  local_feat: (batch, D, N)
    # For each (i, j) pair we need s_{ij} ∈ R^{T×N}
    scores = torch.zeros(batch, batch, device=word_embs.device)

    for i in range(batch):
        # word vectors for image i: (D, T) → (T, D)
        e = word_embs[i].t()   # (T, D)
        # local features for all images: (batch, D, N)
        v = local_feat          # (batch, D, N)

        # similarity: s = e @ v  →  (batch, T, N)
        s = torch.bmm(e.unsqueeze(0).expand(batch, -1, -1), v)  # (batch, T, N)

        # normalise over N for each word (eq. 8)
        s_bar = F.softmax(s, dim=2)   # (batch, T, N)

        # region-context: c = s_bar^T @ e   (eq. 9 attention weights)
        # alpha_{j} = softmax(gamma1 * s_bar_{i,j})
        alpha = F.softmax(gamma1 * s_bar, dim=2)    # (batch, T, N)

        # c_i = sum_j alpha_j * v_j    →  (batch, T, D)
        c = torch.bmm(alpha, v.permute(0, 2, 1))    # (batch, T, D)

        # cosine similarity R(c_i, e_i) for each word (eq. 10 inner term)
        e_expand = e.unsqueeze(0).expand(batch, -1, -1)  # (batch, T, D)
        cos = F.cosine_similarity(c, e_expand, dim=2)     # (batch, T)

        # R(Q, D) = log(sum exp(gamma2 * R(c_i, e_i))) / gamma2
        # We use gamma2=5 same as gamma1 here for the per-word score aggregation
        r = torch.log(torch.clamp(torch.sum(torch.exp(cos), dim=1), min=1e-8))  # (batch,)
        scores[i] = r

    return scores
