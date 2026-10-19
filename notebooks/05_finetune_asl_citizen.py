# Goal 1, step 7: v5 = the v4 model fine-tuned on ASL Citizen for our service-counter vocabulary.
#
# v1-v4 were trained on the Kaggle data only (250 signs, phone videos). ASL Citizen is the main dataset
# (docs/GOAL1_LOG.md, "Dataset roles"): webcam videos of 52 Deaf signers at home, with an official
# signer-independent split (35 train / 6 validation / 11 test signers). This notebook
#   - loads the landmarks that tools/extract_asl_citizen.py saved for the 155 target glosses,
#   - merges gloss variants into one word (WANT1 + WANT2 -> "want": the device speaks the same word),
#   - starts from the v4 weights (same model, same body-coordinate input) with a new output layer; rows
#     for the words that also exist in Kaggle are copied from the v4 output layer,
#   - trains the output layer alone for a few epochs, then the whole model with a small learning rate,
#     the v3/v4 recipe otherwise (augmentation, late dropout, AWP, label smoothing),
#   - picks the best epoch on the validation signers and reports the test signers once.
# Before training it also reports v4 as it is on the ASL Citizen test videos of the words that are in
# both datasets: that is the domain gap the fine-tuning has to close.
#
# Colab (Runtime -> L4 or A100 GPU). Cell 1:
#
#   from google.colab import drive
#   import os
#   drive.mount("/content/drive")
#   D = "/content/drive/MyDrive/sign2speech"
#   !cp {D}/asl_citizen/asl_citizen_landmarks.npy {D}/asl_citizen/asl_citizen_meta.npz {D}/asl_citizen/asl_citizen_info.json /content/
#   os.environ["DATA"] = "/content"
#   os.environ["OUT"] = D
#
# Cell 2: this whole file (or clone the repo and `%run /content/repo/notebooks/05_finetune_asl_citizen.py`).
# Under an hour on an L4. Progress is saved to Drive after every epoch (asl_citizen_v5_last.pt); after a
# disconnect, run both cells again and training continues. Settings via os.environ in cell 1:
#   EPOCHS (100), PATIENCE (30, 0 = never stop early), HEAD_EPOCHS (5: output layer only, first),
#   INIT ("v4", or "scratch" to train the same model from random weights for comparison),
#   PRETRAINED (OUT/islr_model_v4.pt), MERGE_VARIANTS ("1"; "0" keeps WANT1 and WANT2 as separate classes).
#
# Outputs (in OUT): asl_citizen_model_v5.pt (best epoch; loads in src/live_demo.py like v4, no --x-scale),
# asl_citizen_v5_logits.npz and asl_citizen_v5_info.json (for src/calibrate.py and src/confusions.py).
# ASL Citizen may not be redistributed: these stay in the team Drive.
import glob
import json
import math
import os
import re
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
MAX_LEN = 64
DIM = 192
EPOCHS = int(os.environ.get("EPOCHS", 100))
PATIENCE = int(os.environ.get("PATIENCE", 30))
HEAD_EPOCHS = int(os.environ.get("HEAD_EPOCHS", 5))
INIT = os.environ.get("INIT", "v4")
MERGE_VARIANTS = os.environ.get("MERGE_VARIANTS", "1") == "1"
BATCH = 64
LR = 1e-4 if INIT == "v4" else 5e-4   # fine-tuning: 5x smaller than training from scratch
HEAD_LR_MULT = 5 if INIT == "v4" else 1  # the new output layer learns faster than the pretrained layers
WEIGHT_DECAY = 0.1 * LR
R0 = 2.08
LATE_START = 10                        # late dropout and AWP from this epoch (after the head-only epochs)
AWP_DELTA = 0.2
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("OUT", ".")
PRETRAINED = os.environ.get("PRETRAINED", os.path.join(OUT, "islr_model_v4.pt"))
if INIT != "v4":
    HEAD_EPOCHS = 0  # nothing pretrained to protect


def find(name):
    for pattern in [os.path.join(os.environ.get("DATA", ""), name), os.path.join(OUT, "asl_citizen", name)]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found; run tools/extract_asl_citizen.py --merge first")


meta = np.load(find("asl_citizen_meta.npz"))
with open(find("asl_citizen_info.json")) as f:
    info = json.load(f)
landmarks = np.load(find("asl_citizen_landmarks.npy"), mmap_mode="r")
N_POINTS = len(info["keep"])  # 94: lips 0-39, left hand 40-60, upper body 61-72, right hand 73-93
glosses = info["signs"]


def word_of(gloss):
    """ASL Citizen gloss -> the word the device will speak: WANT1, WANT2 -> want."""
    return (re.sub(r"\d+$", "", gloss) if MERGE_VARIANTS else gloss).lower()


words = sorted(set(word_of(g) for g in glosses))
y = np.array([words.index(word_of(g)) for g in glosses])[meta["label"]].astype(np.int64)
N_CLASSES = len(words)
print(f"{len(y)} videos, {len(glosses)} glosses -> {N_CLASSES} classes, {len(info['signers'])} signers, device {device}")

SH_L, SH_R, LIP_ROWS = 61, 62, slice(0, 40)  # pose 11 and 12, and the 40 lip points, among the 94 kept


def canonical(seq):
    """(T, P, 2) raw MediaPipe x, y -> body coordinates: origin between the shoulders, 1 = shoulder width,
    x rescaled so shoulder width / shoulder-to-mouth height = R0 (removes the camera's aspect ratio)."""
    seq = np.array(seq, np.float32)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w = np.abs(seq[:, SH_L, 0] - seq[:, SH_R, 0])
        h = (seq[:, SH_L, 1] + seq[:, SH_R, 1]) / 2 - np.nanmean(seq[:, LIP_ROWS, 1], axis=1)
        ratio = np.nanmedian(np.where(h > 0.02, w / h, np.nan))
        if np.isfinite(ratio) and ratio > 0:
            seq[..., 0] *= np.clip(R0 / ratio, 0.3, 3.0)
        mid = np.nanmedian((seq[:, SH_L] + seq[:, SH_R]) / 2, axis=0)
        width = np.nanmedian(np.linalg.norm(seq[:, SH_L] - seq[:, SH_R], axis=1))
        if not (np.isfinite(mid).all() and np.isfinite(width) and width > 0):  # no body: use the sign's own spread
            mid = np.nanmean(seq.reshape(-1, 2), axis=0)
            width = 4 * np.nanstd(seq.reshape(-1, 2))
            if not (np.isfinite(mid).all() and np.isfinite(width) and width > 0):
                return seq
    return (seq - mid) / width


# ---- every video as a fixed (MAX_LEN, N_POINTS, 2) array in body coordinates; NaN = missing or padding ----
# identical to 04_train_islr_v4.py and SignClassifier.prepare(): drop frames without a hand, resample, canonical()
t0 = time.time()
n = len(y)
X = np.full((n, MAX_LEN, N_POINTS, 2), np.nan, np.float16)
lengths_list = []
for i, (o, l) in enumerate(zip(meta["offset"], meta["length"])):
    seq = landmarks[o:o + l, :, :2]
    has_hand = ~np.isnan(seq[:, 40:61, 0]).all(1) | ~np.isnan(seq[:, 73:94, 0]).all(1)
    if has_hand.any():
        seq = seq[has_hand]
    if len(seq) > MAX_LEN:
        seq = seq[np.linspace(0, len(seq) - 1, MAX_LEN).round().astype(int)]
    X[i, :len(seq)] = canonical(seq)
    lengths_list.append(len(seq))
lengths = np.array(lengths_list)
print(f"arrays ready in {time.time() - t0:.0f} s; frames with a hand per video: median {np.median(lengths):.0f}, "
      f"min {lengths.min()}, {(lengths >= MAX_LEN).mean():.0%} of videos cut to {MAX_LEN}")

# ---- the official ASL Citizen split: signers in train, validation and test do not overlap ----
split = meta["split"]
train_idx, val_idx, test_idx = (np.where(split == k)[0] for k in (0, 1, 2))
signer = meta["participant_id"]
per_word = np.bincount(y[train_idx], minlength=N_CLASSES)
print(f"train {len(train_idx)} videos from {len(np.unique(signer[train_idx]))} signers "
      f"({per_word.min()}-{per_word.max()} per word, median {np.median(per_word):.0f}), "
      f"validation {len(val_idx)} from {len(np.unique(signer[val_idx]))}, test {len(test_idx)} from {len(np.unique(signer[test_idx]))}")

# mirror image: swap the hands and the left/right body points
FLIP = np.arange(N_POINTS)
FLIP[40:61], FLIP[73:94] = np.arange(73, 94), np.arange(40, 61)
FLIP[61:73] = np.arange(61, 73).reshape(-1, 2)[:, ::-1].ravel()  # pose 11<->12, 13<->14, ...
# lips: the mirror partner of each lip point (face-mesh pairs from the 1st-place code; 0, 13, 14, 17 are on the middle line)
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
mirror = {**dict(zip(LLIP, RLIP)), **dict(zip(RLIP, LLIP))}
lips = info["keep"][:40]
FLIP[:40] = [lips.index(mirror.get(p, p)) for p in lips]
FLIP = torch.tensor(FLIP, device=device)


def augment(x, mask):
    """1st-place augmentations on a batch (B, T, P, 2) in body coordinates (origin between the shoulders,
    1 = shoulder width), each sample drawn separately."""
    b, t = x.shape[:2]
    dev = x.device

    def chance(p):
        return torch.rand(b, device=dev) < p

    # speed 0.5-1.5x (p 0.8): resample each sign to a new number of frames
    length = mask.sum(1)
    rate = torch.where(chance(0.8), 0.5 + torch.rand(b, device=dev), torch.ones(b, device=dev))
    new_length = (length * rate).round().clamp(2, t).long()
    j = torch.arange(t, device=dev)[None]
    src = (j * length[:, None] / new_length[:, None]).floor().long()
    src = torch.minimum(src, (length[:, None] - 1).clamp(min=0))
    x = torch.gather(x, 1, src[:, :, None, None].expand_as(x))
    mask = j < new_length[:, None]
    x = torch.where(mask[:, :, None, None], x, torch.nan)

    # mirror (p 0.5)
    flip = chance(0.5)
    x = torch.where(flip[:, None, None, None], x[:, :, FLIP], x)
    x = torch.stack([torch.where(flip[:, None, None], -x[..., 0], x[..., 0]), x[..., 1]], -1)

    # scale 0.8-1.2, extra x stretch 0.85-1.18, shear +-0.15 on x or y, rotate +-30 deg, shift +-0.2 (p 0.75)
    affine = chance(0.75)
    scale = torch.where(affine, 0.8 + 0.4 * torch.rand(b, device=dev), torch.ones(b, device=dev))
    shear = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * 0.3, torch.zeros(b, device=dev))
    on_x = torch.rand(b, device=dev) < 0.5
    shear_x, shear_y = torch.where(on_x, shear, 0), torch.where(on_x, 0, shear)
    angle = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * math.radians(60), torch.zeros(b, device=dev))
    shift = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * 0.4, torch.zeros(b, device=dev))
    stretch = torch.where(affine, torch.exp((torch.rand(b, device=dev) - 0.5) * 2 * math.log(1.18)),
                          torch.ones(b, device=dev))
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(cos), torch.zeros_like(cos)
    shear_m = torch.stack([torch.stack([one, shear_y], -1), torch.stack([shear_x, one], -1)], -2)
    rot_m = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    stretch_m = torch.stack([torch.stack([stretch, zero], -1), torch.stack([zero, one], -1)], -2)
    m = (rot_m @ shear_m @ stretch_m) * scale[:, None, None]
    x = torch.einsum("bij,btpj->btpi", m, x) + shift[:, None, None, None]

    # hide 20-40% of the sign's frames in one stretch (p 0.5)
    length = mask.sum(1).float()
    size = (length * (0.2 + 0.2 * torch.rand(b, device=dev))).long()
    start = (torch.rand(b, device=dev) * (length - size).clamp(min=1)).long()
    hide = chance(0.5)[:, None] & (j >= start[:, None]) & (j < (start + size)[:, None])
    x = torch.where(hide[:, :, None, None], torch.nan, x)

    # hide every point inside a random square of side 0.8-1.6 shoulder widths (p 0.5)
    corner = torch.rand(b, 2, device=dev) * 4 - 2.5
    side = 0.8 + 0.8 * torch.rand(b, device=dev)
    inside = ((x > corner[:, None, None]) & (x < corner[:, None, None] + side[:, None, None, None])).all(-1)
    inside &= chance(0.5)[:, None, None]
    x = torch.where(inside[..., None], torch.nan, x)
    return x, mask


HANDS = ((40, 61), (73, 94))  # left and right hand rows; landmark 0 = wrist, 9 = middle finger base


def features(x):
    """Body coordinates (B, T, P, 2) -> per frame: x, y, motion to the next frame and over two frames,
    each hand relative to its wrist and size, and whether each hand was found. -> (B, T, P*6 + 84 + 2)."""
    dx = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, :1])], 1)
    dx2 = torch.cat([x[:, 2:] - x[:, :-2], torch.zeros_like(x[:, :2])], 1)
    local, found = [], []
    for a, b in HANDS:
        hand = x[:, :, a:b]
        size = (hand[:, :, 9] - hand[:, :, 0]).norm(dim=-1, keepdim=True).clamp(min=1e-3)
        local.append((hand - hand[:, :, :1]) / size[..., None])
        found.append((~torch.isnan(hand[..., 0])).any(-1, keepdim=True).to(x.dtype))
    feats = torch.cat([x, dx, dx2], -1).flatten(2)
    return torch.cat([feats, torch.cat(local, 2).flatten(2)] + found, -1).nan_to_num(0)


N_IN = N_POINTS * 6 + 84 + 2


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


class ConvBlock(nn.Module):
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


class TransformerBlock(nn.Module):
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
    def __init__(self, n_in, n_classes, dim=DIM, late_drop=0.8):
        super().__init__()
        self.stem = nn.Linear(n_in, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.05)
        self.blocks = nn.ModuleList([ConvBlock(dim), ConvBlock(dim), ConvBlock(dim), TransformerBlock(dim),
                                     ConvBlock(dim), ConvBlock(dim), ConvBlock(dim), TransformerBlock(dim)])
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


def batches(idx, shuffle):
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), BATCH):
        j = np.sort(idx[i:i + BATCH])
        x = torch.from_numpy(X[j]).to(device).float()
        mask = torch.arange(MAX_LEN, device=device)[None] < torch.from_numpy(lengths[j]).to(device)[:, None]
        yield x, mask, torch.from_numpy(y[j]).to(device)


def evaluate(net, idx):
    net.eval()
    logits, labels = [], []
    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
        for x, mask, t in batches(idx, shuffle=False):
            logits.append(net(features(x), mask).float())
            labels.append(t)
    logits, labels = torch.cat(logits), torch.cat(labels)
    top5 = logits.topk(min(5, logits.shape[1]), 1).indices
    return logits, labels, (top5[:, 0] == labels).float().mean().item(), (top5 == labels[:, None]).any(1).float().mean().item()


def accuracy_among(logits, labels, cols):
    """Top-1 when the choice is restricted to the classes `cols` (labels are in `cols`)."""
    cols = torch.as_tensor(cols, device=logits.device)
    pred = cols[logits[:, cols].argmax(1)]
    return (pred == labels).float().mean().item()


# ---- the pretrained model, and how well it does on ASL Citizen before any fine-tuning ----
model = SignModelV3(N_IN, N_CLASSES).to(device)
print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, {N_CLASSES} classes")
shared = []  # (class index here, class index in the pretrained model)
if INIT == "v4":
    ck = torch.load(PRETRAINED, map_location=device, weights_only=False)
    assert ck.get("arch") == "v4" and ck["n_in"] == N_IN and list(ck["keep"]) == list(info["keep"]), "need a v4 model"
    shared = [(i, ck["signs"].index(w)) for i, w in enumerate(words) if w in ck["signs"]]
    if shared:
        v4 = SignModelV3(N_IN, ck["n_classes"], ck["dim"]).to(device)
        v4.load_state_dict(ck["state_dict"])
        ours, theirs = zip(*shared)
        sub = test_idx[np.isin(y[test_idx], ours)]
        zt, yt, _, _ = evaluate(v4, sub)
        to_v4 = torch.full((N_CLASSES,), -1, device=device)
        to_v4[list(ours)] = torch.tensor(theirs, device=device)
        base = accuracy_among(zt, to_v4[yt], list(theirs))
        print(f"v4 as it is, on the {len(sub)} ASL Citizen test videos of the {len(shared)} words that are also in Kaggle "
              f"(choice among those {len(shared)}): top-1 {base:.1%}")
        del v4
    state = {k: v for k, v in ck["state_dict"].items() if not k.startswith("head.")}
    missing = model.load_state_dict(state, strict=False).missing_keys
    assert all(k.startswith("head.") for k in missing), missing
    with torch.no_grad():  # output rows of the shared words start from v4's rows, the others from random
        for i, j in shared:
            model.head.weight[i] = ck["state_dict"]["head.weight"][j]
            model.head.bias[i] = ck["state_dict"]["head.bias"][j]
    print(f"loaded {os.path.basename(PRETRAINED)} (epoch {ck.get('epoch')}, Kaggle validation {ck.get('val_top1', 0):.1%}); "
          f"{len(shared)} of {N_CLASSES} output rows copied")
else:
    print("training from random weights (INIT=scratch)")

head_params = list(model.head.parameters())
body_params = [p for n_, p in model.named_parameters() if not n_.startswith("head.")]
try:
    opt = torch.optim.RAdam([{"params": body_params, "lr": LR}, {"params": head_params, "lr": LR * HEAD_LR_MULT}],
                            lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)
except TypeError:  # older PyTorch
    opt = torch.optim.AdamW([{"params": body_params, "lr": LR}, {"params": head_params, "lr": LR * HEAD_LR_MULT}],
                            lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_idx) / BATCH)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps_per_epoch, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=device == "cuda")
params = list(model.parameters())


def loss_of(xa, ma, t):
    with torch.autocast(device, enabled=device == "cuda"):
        out = model(features(xa), ma)
        return F.cross_entropy(out.float(), t, label_smoothing=0.1), out


# ---- resume after a Colab disconnect ----
MODEL_OUT = os.path.join(OUT, "asl_citizen_model_v5.pt")
LAST = os.path.join(OUT, "asl_citizen_v5_last.pt")
start_epoch, best, best_epoch = 1, -1.0, 0
if os.path.exists(LAST) and os.environ.get("RESUME", "1") == "1":
    ck_last = torch.load(LAST, map_location=device, weights_only=False)
    if ck_last.get("init") == INIT and ck_last.get("n_classes") == N_CLASSES:
        model.load_state_dict(ck_last["model"]); opt.load_state_dict(ck_last["opt"]); sched.load_state_dict(ck_last["sched"])
        scaler.load_state_dict(ck_last["scaler"])
        start_epoch, best, best_epoch = ck_last["epoch"] + 1, ck_last["best"], ck_last["best_epoch"]
        print(f"resuming after epoch {ck_last['epoch']} (best validation so far {best:.1%} at epoch {best_epoch})")
    else:
        print(f"ignoring {LAST}: it is from a different setting")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    head_only = epoch <= HEAD_EPOCHS
    for p in body_params:
        p.requires_grad_(not head_only)
    late = epoch > max(LATE_START, HEAD_EPOCHS)
    model.late_drop_on = late
    t0, total, correct, seen = time.time(), 0.0, 0, 0
    for x, mask, t in batches(train_idx, shuffle=True):
        xa, ma = augment(x, mask)
        loss, out = loss_of(xa, ma, t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if late:
            # AWP: push each weight tensor 0.2 along its gradient, compute the update there, then move back
            with torch.no_grad():
                deltas = []
                for p in params:
                    d = None
                    if p.grad is not None:
                        g = p.grad.float()
                        norm = g.norm()
                        if torch.isfinite(norm) and norm > 0:
                            d = (AWP_DELTA * g / norm).to(p.dtype)
                            p.add_(d)
                    deltas.append(d)
            opt.zero_grad(set_to_none=True)
            loss2, _ = loss_of(xa, ma, t)
            scaler.scale(loss2).backward()
            with torch.no_grad():
                for p, d in zip(params, deltas):
                    if d is not None:
                        p.sub_(d)
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item() * len(t)
        correct += (out.argmax(1) == t).sum().item()
        seen += len(t)
    logits, labels, top1, top5 = evaluate(model, val_idx)
    print(f"epoch {epoch:3d}  loss {total / seen:.3f}  train acc {correct / seen:.1%}  "
          f"val top-1 {top1:.1%}  top-5 {top5:.1%}  {time.time() - t0:.0f} s"
          + ("  (output layer only)" if head_only else "") + ("  (late dropout + AWP on)" if late and epoch == max(LATE_START, HEAD_EPOCHS) + 1 else ""))
    if top1 > best:
        best, best_epoch = top1, epoch
        torch.save({"arch": "v4", "state_dict": model.state_dict(), "n_in": N_IN, "n_classes": N_CLASSES,
                    "dim": DIM, "max_len": MAX_LEN, "keep": info["keep"], "signs": words, "glosses": glosses,
                    "merge_variants": MERGE_VARIANTS, "dataset": "asl_citizen", "init": INIT,
                    "pretrained": os.path.basename(PRETRAINED) if INIT == "v4" else None,
                    "preprocess": {"coords": "body", "r0": R0, "drop_frames_without_hands": True,
                                   "resample": "uniform", "x_scale": None},
                    "val_top1": top1, "epoch": epoch}, MODEL_OUT)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                "init": INIT, "n_classes": N_CLASSES}, LAST)
    if PATIENCE and epoch - best_epoch >= PATIENCE:
        print(f"validation has not improved for {PATIENCE} epochs: stopping")
        break

# ---- final test with the best epoch's weights; the 11 test signers were never used to choose anything ----
model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=False)["state_dict"])
val_logits, val_labels, _, _ = evaluate(model, val_idx)
test_logits, test_labels, test_top1, test_top5 = evaluate(model, test_idx)
np.savez(os.path.join(OUT, "asl_citizen_v5_logits.npz"),
         val_logits=val_logits.cpu().numpy(), val_labels=val_labels.cpu().numpy(), val_signer=signer[val_idx],
         test_logits=test_logits.cpu().numpy(), test_labels=test_labels.cpu().numpy(), test_signer=signer[test_idx])
with open(os.path.join(OUT, "asl_citizen_v5_info.json"), "w") as f:
    json.dump({"keep": info["keep"], "signs": words, "glosses": glosses, "signers": info["signers"]}, f)
print(f"best validation top-1: {best:.1%} (epoch {best_epoch})")
print(f"TEST top-1: {test_top1:.1%}  top-5: {test_top5:.1%}  ({len(test_idx)} videos, {N_CLASSES} words, "
      f"{len(np.unique(signer[test_idx]))} signers never seen in training or validation)")
correct = (test_logits.argmax(1) == test_labels).cpu().numpy()
for s in np.unique(signer[test_idx]):
    m = signer[test_idx] == s
    print(f"  test signer {info['signers'][s]}: top-1 {correct[m].mean():.1%} ({m.sum()} videos)")
if shared:
    ours = [i for i, _ in shared]
    m = np.isin(test_labels.cpu().numpy(), ours)
    print(f"the {len(shared)} words also in Kaggle, same test videos as the v4 baseline above (choice among those {len(shared)}): "
          f"v5 top-1 {accuracy_among(test_logits[torch.from_numpy(m).to(device)], test_labels[torch.from_numpy(m).to(device)], ours):.1%}  "
          f"(v4 without fine-tuning: {base:.1%})")
    other = ~np.isin(np.arange(N_CLASSES), ours)
    m2 = np.isin(test_labels.cpu().numpy(), np.where(other)[0])
    print(f"words new to the model ({other.sum()}): top-1 {correct[m2].mean():.1%};  words also in Kaggle ({len(shared)}): top-1 {correct[m].mean():.1%}")
