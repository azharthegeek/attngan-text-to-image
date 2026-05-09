"""
AttnGAN Generator and Discriminator networks.

Architecture (Section 3.1 of the paper):
  - CA_NET          : Conditioning Augmentation
  - ATTENTION        : word-level attention model F_i^attn (eq. 2)
  - ResBlock         : residual block used in generators
  - INIT_STAGE_G     : G0  — noise + sentence → 64×64
  - NEXT_STAGE_G     : G1/G2 — previous hidden + attention → 128×128 / 256×256
  - D_NET64/128/256  : discriminators with Spectral Normalization (our improvement)
  - G_NET            : assembles all generator stages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint as ckpt_fn


# ---------------------------------------------------------------------------
# Conditioning Augmentation (F^ca)
# ---------------------------------------------------------------------------

class CA_NET(nn.Module):
    """
    Maps global sentence vector ē → conditioning vector c via reparameterisation.
    Outputs (c, mu, log_var) where c = mu + eps * sigma  (eps ~ N(0,I)).
    The KL(N(mu,sigma^2) || N(0,I)) term in the generator loss enforces a
    smooth, compact conditioning space.
    """

    def __init__(self, sent_dim, ca_dim):
        super().__init__()
        # fc maps sentence → [mu | log_var], each of size ca_dim
        self.fc = nn.Linear(sent_dim, ca_dim * 2)
        self.relu = nn.ReLU(inplace=True)
        self.ca_dim = ca_dim

    def encode(self, sent):
        x = self.relu(self.fc(sent))
        mu      = x[:, :self.ca_dim]
        log_var = x[:, self.ca_dim:]
        return mu, log_var

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, sent):
        mu, log_var = self.encode(sent)
        c = self.reparameterise(mu, log_var)
        return c, mu, log_var


# ---------------------------------------------------------------------------
# Attention module F_i^attn  (equation 2 in the paper)
# ---------------------------------------------------------------------------

class ATTENTION(nn.Module):
    """
    Computes word-context vectors for each image sub-region.

    Given word features e (D×T) and hidden image features h (D×N):
      1. Project words: e' = U e
      2. Attention weights: beta_{j,i} = softmax(h_j^T e'_i)
      3. Context vector:    c_j = sum_i beta_{j,i} e'_i
    Returns context matrix (D×N) merged with h → (2D×N).
    """

    def __init__(self, idf, cdf):
        """
        Args:
            idf : image feature dimension D
            cdf : text/context feature dimension (same D in this impl.)
        """
        super().__init__()
        self.conv = nn.Conv1d(cdf, idf, 1)   # projects word features to image space

    def forward(self, word_embs, h):
        """
        Args:
            word_embs : (batch, D, T)  — word feature matrix e
            h         : (batch, D, N)  — image hidden features (flattened spatial)
        Returns:
            context   : (batch, 2D, N) — word-context concatenated with h
            attn_map  : (batch, T, N)  — attention weights for visualisation
        """
        # Project words into image feature space
        e_prime = self.conv(word_embs)    # (batch, D, T)

        # Attention weights: score_{j,i} = h_j^T e'_i
        # h: (batch, D, N)  e_prime: (batch, D, T)
        attn = torch.bmm(h.permute(0, 2, 1), e_prime)   # (batch, N, T)
        attn = F.softmax(attn, dim=2)                     # softmax over words
        attn_map = attn.permute(0, 2, 1)                  # (batch, T, N)

        # Context: c_j = sum_i attn_{j,i} * e'_i
        # e_prime: (batch, D, T)  attn: (batch, N, T)  → need (batch, T, N) for bmm
        # result: (batch, D, N)
        context = torch.bmm(e_prime, attn.permute(0, 2, 1))  # (batch, D, N)

        # Concatenate context with original hidden features
        out = torch.cat([context, h], dim=1)             # (batch, 2D, N)
        return out, attn_map.detach()


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Upsampling block used in generators
# ---------------------------------------------------------------------------

def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# G0 — Initial stage: noise + sentence → 64×64
# ---------------------------------------------------------------------------

class INIT_STAGE_G(nn.Module):
    """
    First generator stage.
    Input:  z (batch, Z_DIM)  concatenated with c (batch, CA_DIM)
    Output: hidden features h0 (batch, Ngf, 4, 4) → ... → (batch, Ngf, 64, 64)
            and image x̂0 (batch, 3, 64, 64)
    """

    def __init__(self, z_dim, ca_dim, gf_dim, r_num=2):
        super().__init__()
        in_dim = z_dim + ca_dim       # noise + conditioning
        Ngf = gf_dim * 8             # e.g. 32*8 = 256

        # Project to spatial: (in_dim) → (Ngf * 4 * 4)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, Ngf * 4 * 4 * 4, bias=False),
            nn.BatchNorm1d(Ngf * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.Ngf = Ngf

        # Residual blocks
        res = [ResBlock(Ngf * 4) for _ in range(r_num)]
        self.residual = nn.Sequential(*res)

        # Upsampling: 4×4 → 8 → 16 → 32 → 64
        self.upsample = nn.Sequential(
            up_block(Ngf * 4, Ngf * 4),
            up_block(Ngf * 4, Ngf * 2),
            up_block(Ngf * 2, Ngf),
            up_block(Ngf, Ngf // 2),
        )

        # Final conv to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(Ngf // 2, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)   # (batch, Ngf*4, 4, 4)
        h = self.residual(x)
        h = self.upsample(h)               # (batch, Ngf//2, 64, 64)
        img = self.to_rgb(h)
        return h, img


# ---------------------------------------------------------------------------
# G1/G2 — Next stages: attend to words, upsample ×2
# ---------------------------------------------------------------------------

class NEXT_STAGE_G(nn.Module):
    """
    Subsequent generator stages (G1: 64→128, G2: 128→256).

    Applies the attention model to h from the previous stage, then
    runs residual blocks + upsampling to produce a higher-res image.
    """

    def __init__(self, gf_dim, ef_dim, r_num=2):
        """
        Args:
            gf_dim : base generator filter count
            ef_dim : text embedding dimension D
        """
        super().__init__()
        Ngf = gf_dim // 2   # input hidden channels from previous stage

        # Attention merges context (ef_dim) with h (Ngf) → output (ef_dim + Ngf)
        self.attn = ATTENTION(Ngf, ef_dim)

        # After attention we have 2*Ngf channels; reduce back with 1×1 conv
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(Ngf * 2, Ngf, 1, bias=False),
            nn.BatchNorm2d(Ngf),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        res = [ResBlock(Ngf) for _ in range(r_num)]
        self.residual = nn.Sequential(*res)

        # Upsample ×2
        self.upsample = up_block(Ngf, Ngf // 2)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(Ngf // 2, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, h, word_embs, mask=None):
        """
        Args:
            h         : (batch, C, H, W)  — hidden from previous stage
            word_embs : (batch, D, T)     — word features from text encoder
        Returns:
            h_new  : (batch, C//2, 2H, 2W)
            img    : (batch, 3, 2H, 2W)
            attn   : (batch, T, H*W)
        """
        B, C, H, W = h.shape
        # flatten spatial: (batch, C, H*W)
        h_flat = h.view(B, C, H * W)

        # apply word attention
        context, attn = self.attn(word_embs, h_flat)  # (batch, 2C, H*W)

        # reshape back to spatial
        context = context.view(B, -1, H, W)           # (batch, 2C, H, W)

        # reduce channels
        x = self.channel_reduce(context)              # (batch, C, H, W)
        x = self.residual(x)
        x = self.upsample(x)                          # (batch, C//2, 2H, 2W)
        img = self.to_rgb(x)
        return x, img, attn


# ---------------------------------------------------------------------------
# Full Generator (assembles all stages)
# ---------------------------------------------------------------------------

class G_NET(nn.Module):
    """
    Assembles G0 → G1 → G2.
    Returns a list of (image, attention_map) per stage.
    """

    def __init__(self, z_dim=100, ca_dim=100, gf_dim=32, ef_dim=256, r_num=2,
                 num_stages=3, use_checkpoint=False):
        super().__init__()
        self.ca_net = CA_NET(ef_dim, ca_dim)
        self.num_stages = num_stages
        self.use_checkpoint = use_checkpoint

        # G0: gf_dim*8 hidden channels at 64×64 (Ngf//2 = gf_dim*4)
        self.g0 = INIT_STAGE_G(z_dim, ca_dim, gf_dim, r_num)

        # G1, G2
        self.stages = nn.ModuleList()
        # After G0: hidden has gf_dim//2 channels = gf_dim*4
        # We build NEXT_STAGE_G with gf_dim growing so that Ngf = prev_out_ch
        # G0 output hidden: gf_dim*4 channels (see INIT_STAGE_G.upsample)
        next_gf = gf_dim * 8   # NEXT_STAGE_G expects gf_dim s.t. Ngf=gf_dim//2 = gf_dim*4
        for _ in range(num_stages - 1):
            self.stages.append(NEXT_STAGE_G(next_gf, ef_dim, r_num))
            next_gf = next_gf // 2

    def forward(self, z, sent_emb, word_embs):
        """
        Args:
            z         : (batch, Z_DIM)
            sent_emb  : (batch, D)       — global sentence feature
            word_embs : (batch, D, T)    — word feature matrix
        Returns:
            fake_imgs : list of (batch, 3, size) at each scale
            attn_maps : list of attention maps (one per subsequent stage)
            mu, log_var : for KL loss
        """
        c, mu, log_var = self.ca_net(sent_emb)    # conditioning vector

        if self.use_checkpoint:
            h, img0 = ckpt_fn(self.g0, z, c, use_reentrant=False)
        else:
            h, img0 = self.g0(z, c)
        fake_imgs = [img0]
        attn_maps = []

        for stage in self.stages:
            if self.use_checkpoint:
                h, img, attn = ckpt_fn(stage, h, word_embs, use_reentrant=False)
            else:
                h, img, attn = stage(h, word_embs)
            fake_imgs.append(img)
            attn_maps.append(attn)

        return fake_imgs, attn_maps, mu, log_var


# ---------------------------------------------------------------------------
# Discriminators with Spectral Normalization (our improvement)
# ---------------------------------------------------------------------------
# Spectral normalization enforces a Lipschitz constraint on D,
# preventing gradient explosion and stabilising adversarial training.

def snconv(in_ch, out_ch, kernel=3, stride=1, padding=1):
    """Conv2d wrapped with spectral normalisation."""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False))


class DownBlock(nn.Module):
    """Strided conv + LeakyReLU for discriminator downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            snconv(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CondDiscriminator(nn.Module):
    """
    Shared conditional + unconditional output heads.
    Given spatial features (batch, C, 4, 4) and sentence vector (batch, D):
      - Unconditional head: classify real/fake ignoring text
      - Conditional head  : classify real/fake given sentence match
    """

    def __init__(self, ndf, ef_dim):
        super().__init__()
        # Merge sentence into spatial: project to ndf then tile
        self.cond_proj = nn.Sequential(
            nn.Linear(ef_dim, ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Joint conv on [img_feat | sent_feat] → 1×1
        self.joint_conv = nn.Sequential(
            snconv(ndf + ndf, ndf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(ndf, 1, 4, 1, 0),   # 4×4 → 1×1
        )
        # Unconditional conv on img_feat only
        self.uncond_conv = snconv(ndf, 1, 4, 1, 0)

    def forward(self, h, sent):
        # Unconditional
        uncond = self.uncond_conv(h)                          # (batch, 1, 1, 1)

        # Conditional: tile sentence vector to spatial size of h
        s = self.cond_proj(sent)                              # (batch, ndf)
        s = s.unsqueeze(2).unsqueeze(3)                       # (batch, ndf, 1, 1)
        s = s.expand(h.size(0), s.size(1), h.size(2), h.size(3))  # (batch, ndf, 4, 4)
        joint = torch.cat([h, s], dim=1)                      # (batch, 2*ndf, 4, 4)
        cond = self.joint_conv(joint)                         # (batch, 1, 1, 1)

        return uncond.squeeze(), cond.squeeze()


class D_NET64(nn.Module):
    """Discriminator for 64×64 images (D0)."""

    def __init__(self, df_dim=64, ef_dim=256):
        super().__init__()
        ndf = df_dim
        # 64×64 → 32 → 16 → 8 → 4
        self.encode = nn.Sequential(
            snconv(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(ndf,      ndf * 2),
            DownBlock(ndf * 2,  ndf * 4),
            DownBlock(ndf * 4,  ndf * 8),
        )
        self.heads = CondDiscriminator(ndf * 8, ef_dim)

    def forward(self, x, sent):
        h = self.encode(x)          # (batch, ndf*8, 4, 4)
        return self.heads(h, sent)


class D_NET128(nn.Module):
    """Discriminator for 128×128 images (D1)."""

    def __init__(self, df_dim=64, ef_dim=256):
        super().__init__()
        ndf = df_dim
        # 128×128 → 64 → 32 → 16 → 8 → 4
        self.encode = nn.Sequential(
            snconv(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(ndf,      ndf * 2),
            DownBlock(ndf * 2,  ndf * 4),
            DownBlock(ndf * 4,  ndf * 8),
            DownBlock(ndf * 8,  ndf * 16),
        )
        self.compress = nn.Sequential(
            snconv(ndf * 16, ndf * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.heads = CondDiscriminator(ndf * 8, ef_dim)

    def forward(self, x, sent):
        h = self.encode(x)          # (batch, ndf*16, 4, 4)
        h = self.compress(h)        # (batch, ndf*8, 4, 4)
        return self.heads(h, sent)


class D_NET256(nn.Module):
    """Discriminator for 256×256 images (D2)."""

    def __init__(self, df_dim=64, ef_dim=256):
        super().__init__()
        ndf = df_dim
        # 256×256 → 128 → 64 → 32 → 16 → 8 → 4
        self.encode = nn.Sequential(
            snconv(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(ndf,       ndf * 2),
            DownBlock(ndf * 2,   ndf * 4),
            DownBlock(ndf * 4,   ndf * 8),
            DownBlock(ndf * 8,   ndf * 16),
            DownBlock(ndf * 16,  ndf * 32),
        )
        self.compress = nn.Sequential(
            snconv(ndf * 32, ndf * 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(ndf * 16, ndf * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.heads = CondDiscriminator(ndf * 8, ef_dim)

    def forward(self, x, sent):
        h = self.encode(x)
        h = self.compress(h)
        return self.heads(h, sent)
