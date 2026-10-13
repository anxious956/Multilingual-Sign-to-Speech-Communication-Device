# Goal 1, step 3b: train a first sign classifier on the shrunk Kaggle Google ISLR data.
#
# Runs on Colab (Runtime -> Change runtime type -> T4 GPU) or on Kaggle.
#
# Colab: first run this setup cell once. It copies the output of
# 01_preprocess_islr (about 2 GB) from Kaggle to Google Drive the first time,
# and from Drive to the Colab disk every time. It needs your kaggle.json
# (Kaggle -> Settings -> API -> Create New Token) only for the first download.
# Replace USER/NOTEBOOK with the end of your preprocessing notebook's URL,
# kaggle.com/code/USER/NOTEBOOK.
#
#   from google.colab import drive, files
#   import os
#   drive.mount("/content/drive")
#   DRIVE = "/content/drive/MyDrive/sign2speech"
#   os.makedirs(DRIVE, exist_ok=True)
#   if not os.path.exists(f"{DRIVE}/islr_landmarks.npy"):
#       files.upload()  # choose kaggle.json
#       !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#       !kaggle kernels output USER/NOTEBOOK -p {DRIVE}
#   !cp {DRIVE}/islr_* /content/
#   os.environ["DATA"] = "/content"
#   os.environ["OUT"] = DRIVE
#
# Then paste this whole file into the next cell and run it. The model is
# saved to Drive (sign2speech folder) after every epoch that improves.
#
# Kaggle: add the output of 01_preprocess_islr as input ("Add Input" -> Your
# Work -> Notebooks), choose a GPU, paste this file and run.
#
# Three signers are held out for validation, so the accuracy printed is for
# people the model has never seen. Outputs:
#
#   islr_model.pt        weights of the best epoch and the settings needed to rebuild the model
#   islr_val_logits.npz  model outputs and true labels on the validation signers (for step 5, calibration)
#
# The model is a small 1D-CNN + transformer, the same kind of model as the
# winning Kaggle solution but much smaller, so it can run on the Jetson.
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
MAX_LEN = 64          # longer signs are resampled to 64 frames
DIM = 192             # model width
EPOCHS = int(os.environ.get("EPOCHS", 40))
BATCH = 128
LR = 2e-3
VAL_SIGNERS = 3
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")


def find(name):
    for pattern in [os.path.join(os.environ.get("DATA", ""), name), f"/kaggle/input/*/{name}",
                    f"/kaggle/input/*/*/{name}", f"/kaggle/input/*/*/*/{name}"]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found; add the output of 01_preprocess_islr as input")


meta = np.load(find("islr_meta.npz"))
with open(find("islr_info.json")) as f:
    info = json.load(f)
landmarks = np.load(find("islr_landmarks.npy"), mmap_mode="r")
N_CLASSES = len(info["signs"])
N_POINTS = len(info["keep"])  # 94: lips 0-39, left hand 40-60, upper body 61-72, right hand 73-93
print(f"{len(meta['label'])} signs, {N_CLASSES} classes, {N_POINTS} landmarks, device {device}")

# ---- every sign as a fixed (MAX_LEN, N_POINTS, 2) array of x, y; NaN = missing or padding ----
t0 = time.time()
n = len(meta["label"])
X = np.full((n, MAX_LEN, N_POINTS, 2), np.nan, np.float16)
lengths_list = []
for i, (o, l) in enumerate(zip(meta["offset"], meta["length"])):
    seq = landmarks[o:o + l, :, :2]
    # about 40% of frames have no hand at all; drop them (unless the whole sign has none)
    has_hand = ~np.isnan(seq[:, 40:61, 0]).all(1) | ~np.isnan(seq[:, 73:94, 0]).all(1)
    if has_hand.any():
        seq = seq[has_hand]
    if len(seq) > MAX_LEN:
        seq = seq[np.linspace(0, len(seq) - 1, MAX_LEN).round().astype(int)]
    X[i, :len(seq)] = seq
    lengths_list.append(len(seq))
lengths = np.array(lengths_list)
y = meta["label"].astype(np.int64)
print(f"arrays ready in {time.time() - t0:.0f} s, {X.nbytes / 1e9:.1f} GB")

# ---- hold out a few signers for validation ----
signers = np.unique(meta["participant_id"])
val_signers = rng.choice(signers, VAL_SIGNERS, replace=False)
is_val = np.isin(meta["participant_id"], val_signers)
train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f"train {len(train_idx)} signs, validation {len(val_idx)} signs from signers {sorted(val_signers.tolist())}")

# mirror image: swap the hands and the left/right body points
FLIP = np.arange(N_POINTS)
FLIP[40:61], FLIP[73:94] = np.arange(73, 94), np.arange(40, 61)
FLIP[61:73] = np.arange(61, 73).reshape(-1, 2)[:, ::-1].ravel()  # pose 11<->12, 13<->14, ...
FLIP = torch.tensor(FLIP, device=device)


def augment(x):
    """Random mirror, rotation and stretch of a batch (B, T, P, 2) of raw coordinates."""
    b = x.shape[0]
    flip = torch.rand(b, device=device) < 0.5
    x = torch.where(flip[:, None, None, None], x[:, :, FLIP], x)
    x = torch.stack([torch.where(flip[:, None, None], 1 - x[..., 0], x[..., 0]), x[..., 1]], -1)
    angle = (torch.rand(b, device=device) - 0.5) * math.radians(30)
    sx, sy = 1 + (torch.rand(2, b, device=device) - 0.5) * 0.3
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot = torch.stack([torch.stack([cos * sx, -sin * sx], -1), torch.stack([sin * sy, cos * sy], -1)], -2)
    return torch.einsum("bij,btpj->btpi", rot, x - 0.5) + 0.5


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
    def __init__(self, n_in, n_classes, dim=DIM, blocks=2, drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_in, dim), nn.LayerNorm(dim))
        self.convs = nn.ModuleList([ConvBlock(dim, drop=drop) for _ in range(3 * blocks)])
        self.attn = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, 2 * dim, drop, batch_first=True, norm_first=True)
                                   for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, n_classes))

    def forward(self, x, mask):
        x = self.stem(x)
        for b, attn in enumerate(self.attn):
            for conv in self.convs[3 * b:3 * b + 3]:
                x = conv(x, mask)
            x = attn(x, src_key_padding_mask=~mask)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def batches(idx, shuffle):
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), BATCH):
        j = np.sort(idx[i:i + BATCH])
        x = torch.from_numpy(X[j]).to(device).float()
        mask = torch.arange(MAX_LEN, device=device)[None] < torch.from_numpy(lengths[j]).to(device)[:, None]
        yield x, mask, torch.from_numpy(y[j]).to(device)


model = SignModel(N_POINTS * 4, N_CLASSES).to(device)
print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
steps = EPOCHS * math.ceil(len(train_idx) / BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler(enabled=device == "cuda")


def evaluate():
    model.eval()
    logits, labels = [], []
    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
        for x, mask, t in batches(val_idx, shuffle=False):
            logits.append(model(features(x), mask).float())
            labels.append(t)
    logits, labels = torch.cat(logits), torch.cat(labels)
    top5 = logits.topk(min(5, N_CLASSES), 1).indices
    return logits, labels, (top5[:, 0] == labels).float().mean().item(), (top5 == labels[:, None]).any(1).float().mean().item()


best = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, total, correct, seen = time.time(), 0.0, 0, 0
    for x, mask, t in batches(train_idx, shuffle=True):
        with torch.autocast(device, enabled=device == "cuda"):
            out = model(features(augment(x)), mask)
            loss = F.cross_entropy(out.float(), t, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item() * len(t)
        correct += (out.argmax(1) == t).sum().item()
        seen += len(t)
    logits, labels, top1, top5 = evaluate()
    print(f"epoch {epoch:2d}  loss {total / seen:.3f}  train acc {correct / seen:.1%}  "
          f"val top-1 {top1:.1%}  top-5 {top5:.1%}  {time.time() - t0:.0f} s")
    if top1 > best:
        best = top1
        torch.save({"state_dict": model.state_dict(), "n_in": N_POINTS * 4, "n_classes": N_CLASSES, "dim": DIM,
                    "max_len": MAX_LEN, "keep": info["keep"], "signs": info["signs"], "val_top1": top1},
                   os.path.join(OUT, "islr_model.pt"))
        np.savez(os.path.join(OUT, "islr_val_logits.npz"), logits=logits.cpu().numpy(), labels=labels.cpu().numpy())
print(f"best validation top-1: {best:.1%} (signers not seen in training)")
