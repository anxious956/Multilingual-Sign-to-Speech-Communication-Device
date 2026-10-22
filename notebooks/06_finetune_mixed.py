# Goal 1, step 7b: v6 = the v4 model fine-tuned on ASL Citizen AND Kaggle together.
#
# Why: v5 (notebook 05, ASL Citizen only) reached 86.6% on ASL Citizen's test signers but was worse than v4
# live on our laptop webcam: low confidence everywhere, WATER read as FINE. Training the same model from
# scratch scored almost the same as v5, i.e. fine-tuning on ASL Citizen alone wiped out most of what v4 had
# learned from the 94,000 Kaggle signs, and that is what made v4 work on our camera. Slowing the live clips
# down did not help v5 (src/compare_models.py), so it is not the frame rate.
#
# v6 keeps training on Kaggle while learning the ASL Citizen words:
#   - one output layer for the 134 ASL Citizen words plus the Kaggle signs that are not among them; Kaggle
#     signs that are the same word map onto it (thankyou -> thankyou, and the aliases below),
#   - every epoch = all ASL Citizen training videos + a fresh random draw of Kaggle training signs
#     (KAGGLE_RATIO x as many), from the Kaggle training signers of v4's split only,
#   - starts from v4 with its whole output layer kept, optional freezing of the first blocks,
#   - picks the best epoch on the ASL Citizen validation signers (choosing among the 134 words) and also
#     prints Kaggle validation accuracy, to see that the Kaggle knowledge is not forgotten,
#   - saves a model that only has the 134 ASL Citizen words, so live_demo / compare_models load it like v5.
#
# Colab (Runtime -> L4 or A100 GPU, High-RAM if offered). Cell 1:
#
#   from google.colab import drive
#   import os
#   drive.mount("/content/drive")
#   D = "/content/drive/MyDrive/sign2speech"
#   !cp {D}/asl_citizen/asl_citizen_landmarks.npy {D}/asl_citizen/asl_citizen_meta.npz {D}/asl_citizen/asl_citizen_info.json /content/
#   !cp {D}/islr_landmarks.npy {D}/islr_meta.npz {D}/islr_info.json /content/
#   os.environ["DATA"] = "/content"
#   os.environ["OUT"] = D
#   !rm -rf /content/repo && git clone -q -b claude/great-hamilton-9vy4ph https://github.com/anxious956/Multilingual-Sign-to-Speech-Communication-Device.git /content/repo
#
# Cell 2: %run /content/repo/notebooks/06_finetune_mixed.py
# About 20-30 minutes on an L4. Resumable (asl_citizen_v6_last.pt). Settings via os.environ in cell 1:
#   EPOCHS (60), PATIENCE (20), KAGGLE_RATIO (1.0: as many Kaggle signs as ASL Citizen videos per epoch),
#   FREEZE_BLOCKS (0: train everything; 1-8 freezes the input layer and the first N of the 8 blocks),
#   HEAD_EPOCHS (3), PRETRAINED (OUT/islr_model_v4.pt), TAG ("v6": output file names).
#
# Outputs (in OUT): asl_citizen_model_{TAG}.pt (134 words, loads like v5), asl_citizen_{TAG}_logits.npz and
# asl_citizen_{TAG}_info.json. ASL Citizen and Kaggle data may not be redistributed: these stay in the Drive.
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
EPOCHS = int(os.environ.get("EPOCHS", 60))
PATIENCE = int(os.environ.get("PATIENCE", 20))
HEAD_EPOCHS = int(os.environ.get("HEAD_EPOCHS", 3))
KAGGLE_RATIO = float(os.environ.get("KAGGLE_RATIO", 1.0))
FREEZE_BLOCKS = int(os.environ.get("FREEZE_BLOCKS", 0))
TAG = os.environ.get("TAG", "v6")
BATCH = 64
LR = 1e-4
HEAD_LR_MULT = 5
WEIGHT_DECAY = 0.1 * LR
R0 = 2.08
LATE_START = 10
AWP_DELTA = 0.2
KAGGLE_VAL_SAMPLE = 3000  # Kaggle validation signs checked every epoch (fixed random subset)
# Kaggle signs that are the same sign and word as an ASL Citizen word under another name
ALIAS = {"minemy": "my", "callonphone": "phone", "food": "eat"}
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("OUT", ".")
PRETRAINED = os.environ.get("PRETRAINED", os.path.join(OUT, "islr_model_v4.pt"))


def find(name):
    for pattern in [os.path.join(os.environ.get("DATA", ""), name), os.path.join(OUT, name),
                    os.path.join(OUT, "asl_citizen", name)]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found; copy it to DATA (see the top of this file)")


cmeta = np.load(find("asl_citizen_meta.npz"))
with open(find("asl_citizen_info.json")) as f:
    cinfo = json.load(f)
clm = np.load(find("asl_citizen_landmarks.npy"), mmap_mode="r")
kmeta = np.load(find("islr_meta.npz"))
with open(find("islr_info.json")) as f:
    kinfo = json.load(f)
klm = np.load(find("islr_landmarks.npy"), mmap_mode="r")
assert list(cinfo["keep"]) == list(kinfo["keep"]), "both datasets must keep the same 94 landmarks"
info = cinfo
N_POINTS = len(info["keep"])  # 94: lips 0-39, left hand 40-60, upper body 61-72, right hand 73-93

# ---- classes: the 134 ASL Citizen words first, then the Kaggle signs that are not one of them ----
words = sorted(set(re.sub(r"\d+$", "", g).lower() for g in cinfo["signs"]))
N_WORDS = len(words)
kaggle_word = {s: ALIAS.get(s, s) for s in kinfo["signs"]}
extra = sorted(s for s, w in kaggle_word.items() if w not in words)
classes = words + extra
N_CLASSES = len(classes)
kaggle_class = np.array([classes.index(kaggle_word[s]) if kaggle_word[s] in words else classes.index(s)
                         for s in kinfo["signs"]])
shared_words = sorted(w for w in kaggle_word.values() if w in words)
print(f"{N_WORDS} ASL Citizen words + {len(extra)} other Kaggle signs = {N_CLASSES} classes; "
      f"{len(shared_words)} words in both datasets; device {device}")

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



def prepare(lm, offsets, lens):
    """Every clip as a fixed (MAX_LEN, N_POINTS, 2) array in body coordinates; NaN = missing or padding.
    Identical to notebooks 04/05 and SignClassifier.prepare(): drop frames without a hand, resample, canonical()."""
    X = np.full((len(offsets), MAX_LEN, N_POINTS, 2), np.nan, np.float16)
    L = np.zeros(len(offsets), np.int64)
    for i, (o, l) in enumerate(zip(offsets, lens)):
        seq = lm[o:o + l, :, :2]
        has_hand = ~np.isnan(seq[:, 40:61, 0]).all(1) | ~np.isnan(seq[:, 73:94, 0]).all(1)
        if has_hand.any():
            seq = seq[has_hand]
        if len(seq) > MAX_LEN:
            seq = seq[np.linspace(0, len(seq) - 1, MAX_LEN).round().astype(int)]
        X[i, :len(seq)] = canonical(seq)
        L[i] = len(seq)
    return X, L


t0 = time.time()
Xc, Lc = prepare(clm, cmeta["offset"], cmeta["length"])
yc = np.array([words.index(re.sub(r"\d+$", "", g).lower()) for g in cinfo["signs"]])[cmeta["label"]]
Xk, Lk = prepare(klm, kmeta["offset"], kmeta["length"])
yk = kaggle_class[kmeta["label"]]
X = np.concatenate([Xc, Xk])
lengths = np.concatenate([Lc, Lk])
y = np.concatenate([yc, yk]).astype(np.int64)
del Xc, Xk
NC = len(yc)  # ASL Citizen clips are rows 0..NC-1, Kaggle clips follow
print(f"arrays ready in {time.time() - t0:.0f} s: {NC} ASL Citizen + {len(yk)} Kaggle clips, {X.nbytes / 1e9:.1f} GB")

# ---- splits. ASL Citizen: official signer split. Kaggle: the same signers held out as in notebook 04 ----
csplit, csigner = cmeta["split"], cmeta["participant_id"]
train_c, val_c, test_c = (np.where(csplit == k)[0] for k in (0, 1, 2))
ksigners = np.unique(kmeta["participant_id"])
held_out = np.random.default_rng(SEED).choice(ksigners, 6, replace=False)  # = 04's draw (its first rng call)
k_is_val = np.isin(kmeta["participant_id"], held_out[:3])
train_k = NC + np.where(~np.isin(kmeta["participant_id"], held_out))[0]
val_k = NC + np.where(k_is_val)[0]
val_k_sample = np.sort(rng.choice(val_k, min(KAGGLE_VAL_SAMPLE, len(val_k)), replace=False))
n_kaggle_per_epoch = int(round(KAGGLE_RATIO * len(train_c)))
print(f"ASL Citizen: train {len(train_c)}, validation {len(val_c)}, test {len(test_c)} videos; "
      f"Kaggle: {len(train_k)} training signs ({n_kaggle_per_epoch} drawn per epoch), "
      f"{len(val_k_sample)} validation signs checked per epoch")

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


def logits_of(net, idx):
    net.eval()
    out, labels = [], []
    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
        for x, mask, t in batches(idx, shuffle=False):
            out.append(net(features(x), mask).float())
            labels.append(t)
    return torch.cat(out), torch.cat(labels)


KAGGLE_COLS = torch.tensor(sorted(set(kaggle_class.tolist())), device=device)


def scores(logits, labels, cols):
    """Top-1 and top-5 when the choice is restricted to the classes `cols` (all labels are in `cols`)."""
    sub = logits[:, cols]
    top = cols[sub.topk(min(5, len(cols)), 1).indices]
    return (top[:, 0] == labels).float().mean().item(), (top == labels[:, None]).any(1).float().mean().item()


WORD_COLS = torch.arange(N_WORDS, device=device)

# ---- start from v4: everything, including its output rows (moved to the merged class order) ----
ck = torch.load(PRETRAINED, map_location=device, weights_only=False)
assert ck.get("arch") == "v4" and ck["n_in"] == N_IN and list(ck["keep"]) == list(info["keep"]), "need a v4 model"
assert list(ck["signs"]) == list(kinfo["signs"]), "v4 must have been trained on this islr_info.json"
v4 = SignModelV3(N_IN, ck["n_classes"], ck["dim"]).to(device)
v4.load_state_dict(ck["state_dict"])
in_kaggle = [words.index(w) for w in shared_words]
sub = test_c[np.isin(y[test_c], in_kaggle)]
zt, yt = logits_of(v4, sub)
to_v4 = {c: j for j, c in enumerate(kaggle_class.tolist()) if c < N_WORDS}
base_shared = scores(zt, torch.tensor([to_v4[int(c)] for c in yt], device=device),
                     torch.tensor(sorted(to_v4[c] for c in in_kaggle), device=device))[0]
zk, _ = logits_of(v4, val_k_sample)  # scored against v4's own 250 classes
base_kaggle = (zk.argmax(1).cpu().numpy() == kmeta["label"][val_k_sample - NC]).mean()
print(f"v4 as it is: ASL Citizen test videos of the {len(in_kaggle)} shared words (choice among them) {base_shared:.1%}; "
      f"Kaggle validation signs (250 classes) {base_kaggle:.1%}")
del v4

model = SignModelV3(N_IN, N_CLASSES).to(device)
state = {k: v for k, v in ck["state_dict"].items() if not k.startswith("head.")}
assert all(k.startswith("head.") for k in model.load_state_dict(state, strict=False).missing_keys)
with torch.no_grad():
    for j, c in enumerate(kaggle_class.tolist()):  # aliases: the last Kaggle sign mapped onto a word wins
        model.head.weight[c] = ck["state_dict"]["head.weight"][j]
        model.head.bias[c] = ck["state_dict"]["head.bias"][j]
print(f"loaded {os.path.basename(PRETRAINED)} (epoch {ck.get('epoch')}); output rows of all {ck['n_classes']} Kaggle signs kept, "
      f"{N_WORDS - len(in_kaggle)} new rows for ASL Citizen words")

frozen = [model.stem, model.stem_bn] + list(model.blocks[:FREEZE_BLOCKS]) if FREEZE_BLOCKS > 0 else []
for m in frozen:
    for p in m.parameters():
        p.requires_grad_(False)
if frozen:
    print(f"frozen: input layer and the first {FREEZE_BLOCKS} of {len(model.blocks)} blocks")
head_params = list(model.head.parameters())
body_params = [p for n_, p in model.named_parameters() if not n_.startswith("head.") and p.requires_grad]
try:
    opt = torch.optim.RAdam([{"params": body_params, "lr": LR}, {"params": head_params, "lr": LR * HEAD_LR_MULT}],
                            lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)
except TypeError:  # older PyTorch
    opt = torch.optim.AdamW([{"params": body_params, "lr": LR}, {"params": head_params, "lr": LR * HEAD_LR_MULT}],
                            lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil((len(train_c) + n_kaggle_per_epoch) / BATCH)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps_per_epoch, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=device == "cuda")
params = [p for p in model.parameters() if p.requires_grad]


def loss_of(xa, ma, t):
    with torch.autocast(device, enabled=device == "cuda"):
        out = model(features(xa), ma)
        return F.cross_entropy(out.float(), t, label_smoothing=0.1), out


def save_words_only(path, val_top1, epoch):
    """The model with only the 134 ASL Citizen word outputs, in the format SignClassifier loads (like v5)."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["head.weight"], sd["head.bias"] = sd["head.weight"][:N_WORDS], sd["head.bias"][:N_WORDS]
    torch.save({"arch": "v4", "state_dict": sd, "n_in": N_IN, "n_classes": N_WORDS, "dim": DIM, "max_len": MAX_LEN,
                "keep": info["keep"], "signs": words, "glosses": cinfo["signs"], "dataset": "asl_citizen+kaggle",
                "init": "v4", "pretrained": os.path.basename(PRETRAINED), "kaggle_ratio": KAGGLE_RATIO,
                "freeze_blocks": FREEZE_BLOCKS, "aliases": ALIAS,
                "preprocess": {"coords": "body", "r0": R0, "drop_frames_without_hands": True,
                               "resample": "uniform", "x_scale": None},
                "val_top1": val_top1, "epoch": epoch}, path)


MODEL_OUT = os.path.join(OUT, f"asl_citizen_model_{TAG}.pt")
LAST = os.path.join(OUT, f"asl_citizen_{TAG}_last.pt")
start_epoch, best, best_epoch = 1, -1.0, 0
if os.path.exists(LAST) and os.environ.get("RESUME", "1") == "1":
    cl = torch.load(LAST, map_location=device, weights_only=False)
    model.load_state_dict(cl["model"]); opt.load_state_dict(cl["opt"]); sched.load_state_dict(cl["sched"])
    scaler.load_state_dict(cl["scaler"])
    start_epoch, best, best_epoch = cl["epoch"] + 1, cl["best"], cl["best_epoch"]
    print(f"resuming after epoch {cl['epoch']} (best validation so far {best:.1%} at epoch {best_epoch})")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    for m in frozen:
        m.eval()  # frozen layers also keep their BatchNorm statistics
    head_only = epoch <= HEAD_EPOCHS
    for p in body_params:
        p.requires_grad_(not head_only)
    late = epoch > max(LATE_START, HEAD_EPOCHS)
    model.late_drop_on = late
    idx = np.concatenate([train_c, rng.choice(train_k, n_kaggle_per_epoch, replace=False)])
    t0, total, correct, seen = time.time(), 0.0, 0, 0
    for x, mask, t in batches(idx, shuffle=True):
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
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item() * len(t)
        correct += (out.argmax(1) == t).sum().item()
        seen += len(t)
    zv, yv = logits_of(model, val_c)
    top1, top5 = scores(zv, yv, WORD_COLS)
    zk, yk_v = logits_of(model, val_k_sample)
    k_top1 = scores(zk, yk_v, KAGGLE_COLS)[0]
    print(f"epoch {epoch:3d}  loss {total / seen:.3f}  train acc {correct / seen:.1%}  "
          f"ASL Citizen val top-1 {top1:.1%}  top-5 {top5:.1%}  |  Kaggle val top-1 {k_top1:.1%}  {time.time() - t0:.0f} s"
          + ("  (output layer only)" if head_only else "") + ("  (late dropout + AWP on)" if late and epoch == max(LATE_START, HEAD_EPOCHS) + 1 else ""))
    if top1 > best:
        best, best_epoch = top1, epoch
        save_words_only(MODEL_OUT, top1, epoch)
        torch.save(model.state_dict(), os.path.join(OUT, f"asl_citizen_{TAG}_best_full.pt"))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch}, LAST)
    if PATIENCE and epoch - best_epoch >= PATIENCE:
        print(f"validation has not improved for {PATIENCE} epochs: stopping")
        break

# ---- final test with the best epoch's weights; the 11 test signers were never used to choose anything ----
model.load_state_dict(torch.load(os.path.join(OUT, f"asl_citizen_{TAG}_best_full.pt"), map_location=device, weights_only=False))
zv, yv = logits_of(model, val_c)
zt, yt = logits_of(model, test_c)
test_top1, test_top5 = scores(zt, yt, WORD_COLS)
np.savez(os.path.join(OUT, f"asl_citizen_{TAG}_logits.npz"),
         val_logits=zv[:, :N_WORDS].cpu().numpy(), val_labels=yv.cpu().numpy(), val_signer=csigner[val_c],
         test_logits=zt[:, :N_WORDS].cpu().numpy(), test_labels=yt.cpu().numpy(), test_signer=csigner[test_c])
with open(os.path.join(OUT, f"asl_citizen_{TAG}_info.json"), "w") as f:
    json.dump({"keep": info["keep"], "signs": words, "glosses": cinfo["signs"], "signers": cinfo["signers"]}, f)
zk, yk_v = logits_of(model, val_k)
base_kaggle_full = base_kaggle  # v4 was scored on the per-epoch sample
print(f"best ASL Citizen validation top-1: {best:.1%} (epoch {best_epoch})")
print(f"TEST top-1: {test_top1:.1%}  top-5: {test_top5:.1%}  ({len(test_c)} videos, {N_WORDS} words, "
      f"{len(np.unique(csigner[test_c]))} signers never seen in training or validation)   [v5: 86.6% / 98.2%]")
pred = WORD_COLS[zt[:, :N_WORDS].argmax(1)].cpu().numpy()
for s in np.unique(csigner[test_c]):
    m = csigner[test_c] == s
    print(f"  test signer {cinfo['signers'][s]}: top-1 {(pred[m] == yt.cpu().numpy()[m]).mean():.1%} ({m.sum()} videos)")
m = torch.from_numpy(np.isin(yt.cpu().numpy(), in_kaggle)).to(device)
print(f"shared words, choice among them: {scores(zt[m], yt[m], torch.tensor(in_kaggle, device=device))[0]:.1%} "
      f"(v4 without fine-tuning {base_shared:.1%}, v5 97.3%)")
print(f"Kaggle validation (3 signers, all {len(val_k)} signs, choice among the Kaggle signs): "
      f"{scores(zk, yk_v, KAGGLE_COLS)[0]:.1%}  (v4 on the per-epoch sample: {base_kaggle_full:.1%}; much lower = Kaggle knowledge forgotten)")
