"""The sign classifier and its input preparation, for running a trained model.

SignModel and features() are copied from notebooks/02_train_islr.py (v1, v2),
SignModelV3 and features_v3() from notebooks/03_train_islr_v3.py; they must stay
identical to those files. SignClassifier picks the right one from the saved model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HANDS = (slice(40, 61), slice(73, 94))  # left and right hand rows among the 94 landmarks kept


def features(x):
    """Normalise each sign (centre and scale) and add frame-to-frame motion. (B, T, P, 2) -> (B, T, P*4)."""
    valid = ~torch.isnan(x)
    count = valid.sum((1, 2), keepdim=True).clamp(min=1)
    mean = torch.where(valid, x, 0).sum((1, 2), keepdim=True) / count
    centred = torch.where(valid, x - mean, 0)
    std = ((centred ** 2).sum((1, 2, 3), keepdim=True) / (2 * count[..., :1])).sqrt().clamp(min=1e-6)
    x = (x - mean) / std
    motion = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, :1])], 1)
    feats = torch.cat([x, motion], -1).nan_to_num(0)
    return feats.flatten(2)


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel=17, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim)
        self.conv = nn.Conv1d(2 * dim, 2 * dim, kernel, padding=kernel // 2, groups=2 * dim)
        self.bn = nn.BatchNorm1d(2 * dim)
        self.project = nn.Linear(2 * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = F.silu(self.expand(self.norm(x))) * mask[..., None]
        h = F.silu(self.bn(self.conv(h.transpose(1, 2)))).transpose(1, 2)
        return x + self.drop(self.project(h))


class SignModel(nn.Module):
    def __init__(self, n_in, n_classes, dim=192, blocks=2, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_in, dim), nn.LayerNorm(dim))
        self.convs = nn.ModuleList([ConvBlock(dim, drop=drop) for _ in range(3 * blocks)])
        self.attn = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, 2 * dim, drop, batch_first=True, norm_first=True)
                                   for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.4), nn.Linear(dim, n_classes))

    def forward(self, x, mask):
        x = self.stem(x)
        for b, attn in enumerate(self.attn):
            for conv in self.convs[3 * b:3 * b + 3]:
                x = conv(x, mask)
            x = attn(x, src_key_padding_mask=~mask)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


# ---- v3 (03_train_islr_v3.py) ----

def features_v3(x):
    """Centre and scale each sign, then x, y, motion to the next frame and over two frames. -> (B, T, P*6)."""
    valid = ~torch.isnan(x)
    count = valid.sum((1, 2), keepdim=True).clamp(min=1)
    mean = torch.where(valid, x, 0).sum((1, 2), keepdim=True) / count
    centred = torch.where(valid, x - mean, 0)
    std = ((centred ** 2).sum((1, 2, 3), keepdim=True) / (2 * count[..., :1])).sqrt().clamp(min=1e-6)
    x = (x - mean) / std
    dx = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, :1])], 1)
    dx2 = torch.cat([x[:, 2:] - x[:, :-2], torch.zeros_like(x[:, :2])], 1)
    return torch.cat([x, dx, dx2], -1).nan_to_num(0).flatten(2)


class DropPath(nn.Module):
    """Drop a whole residual branch for some samples (Keras Dropout with noise_shape (None, 1, 1))."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class ECA(nn.Module):
    """Efficient channel attention: reweight channels from their average over the sign."""

    def __init__(self, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x, mask):
        m = mask[..., None].to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return x * torch.sigmoid(self.conv(pooled[:, None]))[:, 0][:, None]


class ConvBlockV3(nn.Module):
    def __init__(self, dim, kernel=17, drop=0.2):
        super().__init__()
        self.kernel = kernel
        self.expand = nn.Linear(dim, 2 * dim)
        self.conv = nn.Conv1d(2 * dim, 2 * dim, kernel, groups=2 * dim, bias=False)
        self.bn = nn.BatchNorm1d(2 * dim, momentum=0.05)
        self.eca = ECA()
        self.project = nn.Linear(2 * dim, dim)
        self.drop = DropPath(drop)

    def forward(self, x, mask):
        h = F.silu(self.expand(x)) * mask[..., None]
        h = F.pad(h.transpose(1, 2), (self.kernel - 1, 0))  # causal: each frame sees only the past
        h = self.bn(self.conv(h)).transpose(1, 2)
        return x + self.drop(self.project(self.eca(h, mask)))


class TransformerBlockV3(nn.Module):
    def __init__(self, dim, heads=4, expand=2, drop=0.2):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim, momentum=0.05)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.2, batch_first=True)
        self.norm2 = nn.BatchNorm1d(dim, momentum=0.05)
        self.ff = nn.Sequential(nn.Linear(dim, expand * dim, bias=False), nn.SiLU(), nn.Linear(expand * dim, dim, bias=False))
        self.drop1, self.drop2 = DropPath(drop), DropPath(drop)

    def forward(self, x, mask):
        h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        h = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)[0]
        x = x + self.drop1(h)
        return x + self.drop2(self.ff(self.norm2(x.transpose(1, 2)).transpose(1, 2)))


class SignModelV3(nn.Module):
    def __init__(self, n_in, n_classes, dim=192, late_drop=0.8):
        super().__init__()
        self.stem = nn.Linear(n_in, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.05)
        self.blocks = nn.ModuleList([ConvBlockV3(dim), ConvBlockV3(dim), ConvBlockV3(dim), TransformerBlockV3(dim),
                                     ConvBlockV3(dim), ConvBlockV3(dim), ConvBlockV3(dim), TransformerBlockV3(dim)])
        self.top = nn.Linear(dim, 2 * dim)
        self.late_drop = nn.Dropout(late_drop)
        self.late_drop_on = False
        self.head = nn.Linear(2 * dim, n_classes)

    def forward(self, x, mask):
        x = self.stem_bn(self.stem(x).transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        x = self.top(x)
        m = mask[..., None].to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        if self.late_drop_on:
            pooled = self.late_drop(pooled)
        return self.head(pooled)


class SignClassifier:
    """Loads islr_model.pt and classifies one sign given as (frames, 543, 3) landmarks."""

    def __init__(self, path, x_scale=1.0):
        """x_scale stretches x to match the Kaggle videos' proportions (see check_aspect.py)."""
        self.x_scale = x_scale
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("arch") == "v3":
            self.model, self.features = SignModelV3(ckpt["n_in"], ckpt["n_classes"], ckpt["dim"]), features_v3
        else:
            self.model, self.features = SignModel(ckpt["n_in"], ckpt["n_classes"], ckpt["dim"]), features
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.keep, self.signs, self.max_len = ckpt["keep"], ckpt["signs"], ckpt["max_len"]

    def prepare(self, frames):
        """Same steps as the training notebook: keep 94 landmarks (x, y), drop frames without hands, resample."""
        seq = np.asarray(frames, np.float32)[:, self.keep, :2].copy()
        seq[..., 0] *= self.x_scale  # only proportions matter: each sign is centred and scaled later
        has_hand = np.zeros(len(seq), bool)
        for h in HANDS:
            has_hand |= ~np.isnan(seq[:, h, 0]).all(1)
        if has_hand.any():
            seq = seq[has_hand]
        if len(seq) > self.max_len:
            seq = seq[np.linspace(0, len(seq) - 1, self.max_len).round().astype(int)]
        x = np.full((1, self.max_len, len(self.keep), 2), np.nan, np.float32)
        x[0, :len(seq)] = seq
        mask = torch.arange(self.max_len)[None] < len(seq)
        return torch.from_numpy(x), mask

    def limit_to(self, words):
        """Only consider these signs (e.g. the service-counter vocabulary); the rest are never predicted."""
        unknown = [w for w in words if w not in self.signs]
        if unknown:
            raise SystemExit(f"not among the model's signs: {unknown}\navailable: {' '.join(sorted(self.signs))}")
        self.allowed = torch.tensor([s in words for s in self.signs])

    @torch.no_grad()
    def predict(self, frames, top=3):
        """Return the top signs as [(sign, probability), ...]."""
        x, mask = self.prepare(frames)
        logits = self.model(self.features(x), mask)[0]
        if getattr(self, "allowed", None) is not None:
            logits = logits.masked_fill(~self.allowed, -torch.inf)
        probs = logits.softmax(-1)
        top = min(top, int(torch.isfinite(logits).sum()))
        p, i = probs.topk(top)
        return [(self.signs[k], float(v)) for v, k in zip(p, i)]
