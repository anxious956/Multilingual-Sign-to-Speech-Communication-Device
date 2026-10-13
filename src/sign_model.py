"""The sign classifier and its input preparation, for running a trained model.

The model and features() are copied from notebooks/02_train_islr.py and must
stay identical to it; a model trained there is loaded with load_model().
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


class SignClassifier:
    """Loads islr_model.pt and classifies one sign given as (frames, 543, 3) landmarks."""

    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model = SignModel(ckpt["n_in"], ckpt["n_classes"], ckpt["dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.keep, self.signs, self.max_len = ckpt["keep"], ckpt["signs"], ckpt["max_len"]

    def prepare(self, frames):
        """Same steps as the training notebook: keep 94 landmarks (x, y), drop frames without hands, resample."""
        seq = np.asarray(frames, np.float32)[:, self.keep, :2]
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

    @torch.no_grad()
    def predict(self, frames, top=3):
        """Return the top signs as [(sign, probability), ...]."""
        x, mask = self.prepare(frames)
        probs = self.model(features(x), mask).softmax(-1)[0]
        p, i = probs.topk(top)
        return [(self.signs[k], float(v)) for v, k in zip(p, i)]
