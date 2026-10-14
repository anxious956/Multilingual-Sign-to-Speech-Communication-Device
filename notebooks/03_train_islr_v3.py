# Goal 1, step 3c: v3 of the sign classifier, following the Kaggle ISLR 1st-place recipe
# (Hoyeol Sohn, github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution).
#
# Same data, same signer split (seed 42) and same 1D-CNN + transformer size as v2
# (02_train_islr.py), so the results can be compared directly. What is new, all from the 1st-place code:
#   - blocks as in the winner: causal depthwise conv + efficient channel attention (ECA), and
#     whole-branch dropout per sample (Keras Dropout with noise_shape (None, 1, 1))
#   - input: x, y, motion to the next frame and motion over two frames
#   - augmentation: speed 0.5-1.5x (p 0.8), mirror (0.5), scale / shear / rotate +-30 deg / shift (0.75),
#     a 20-40% stretch of time hidden (0.5), a random square of the image hidden (0.5)
#   - "late dropout": 80% dropout before the classifier, switched on after epoch 15
#   - AWP (adversarial weight perturbation) after epoch 15: each step the weights are pushed 0.2
#     along their gradient and the update is computed there, so the model prefers flat minima
#   - RAdam, learning rate 5e-4 per 64 samples, decoupled weight decay 0.1, cosine decay, 300 epochs
# Differences from the winner: 64 frames instead of 384 (faster), our 94 landmarks (lips, hands,
# upper body instead of nose and eyes), one model instead of an ensemble of four.
#
# Colab: run the setup cell first (Runtime -> T4 or L4 GPU):
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
#   !cp -n /content/islr_meta.zip /content/islr_meta.npz 2>/dev/null
#   os.environ["DATA"] = "/content"
#   os.environ["OUT"] = DRIVE
#
#
# then paste this whole file into the next cell. About 2-3 hours on an L4. Progress is saved to
# Drive after every epoch (islr_v3_last.pt); if Colab disconnects, run both cells again and
# training continues where it stopped. Set os.environ["EPOCHS"] before running to change 300.
#
# Outputs (in OUT): islr_model_v3.pt (best epoch), islr_v3_logits.npz (validation and test outputs).
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
MAX_LEN = 64
DIM = 192
EPOCHS = int(os.environ.get("EPOCHS", 300))
BATCH = 128
LR = 5e-4 * BATCH / 64
WEIGHT_DECAY = 0.1
LATE_START = 15          # epoch from which late dropout and AWP are on
AWP_DELTA = 0.2
PATIENCE = int(os.environ.get("PATIENCE", 60))  # stop if validation has not improved for this many epochs
VAL_SIGNERS = 3
TEST_SIGNERS = 3
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
held_out = rng.choice(signers, VAL_SIGNERS + TEST_SIGNERS, replace=False)
val_signers, test_signers = held_out[:VAL_SIGNERS], held_out[VAL_SIGNERS:]
is_val = np.isin(meta["participant_id"], val_signers)
is_test = np.isin(meta["participant_id"], test_signers)
train_idx = np.where(~is_val & ~is_test)[0]
val_idx, test_idx = np.where(is_val)[0], np.where(is_test)[0]
print(f"train {len(train_idx)} signs from {len(signers) - len(held_out)} signers, "
      f"validation {len(val_idx)} from {sorted(val_signers.tolist())}, test {len(test_idx)} from {sorted(test_signers.tolist())}")

# mirror image: swap the hands and the left/right body points
FLIP = np.arange(N_POINTS)
FLIP[40:61], FLIP[73:94] = np.arange(73, 94), np.arange(40, 61)
FLIP[61:73] = np.arange(61, 73).reshape(-1, 2)[:, ::-1].ravel()  # pose 11<->12, 13<->14, ...
FLIP = torch.tensor(FLIP, device=device)




def augment(x, mask):
    """1st-place augmentations on a batch (B, T, P, 2) of raw coordinates, each sample drawn separately."""
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
    x = torch.stack([torch.where(flip[:, None, None], 1 - x[..., 0], x[..., 0]), x[..., 1]], -1)

    # scale 0.8-1.2, shear +-0.15 on x or y, rotate +-30 deg, shift +-0.1 (p 0.75)
    affine = chance(0.75)
    scale = torch.where(affine, 0.8 + 0.4 * torch.rand(b, device=dev), torch.ones(b, device=dev))
    shear = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * 0.3, torch.zeros(b, device=dev))
    on_x = torch.rand(b, device=dev) < 0.5
    shear_x, shear_y = torch.where(on_x, shear, 0), torch.where(on_x, 0, shear)
    angle = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * math.radians(60), torch.zeros(b, device=dev))
    shift = torch.where(affine, (torch.rand(b, device=dev) - 0.5) * 0.2, torch.zeros(b, device=dev))
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(cos), torch.zeros_like(cos)
    shear_m = torch.stack([torch.stack([one, shear_y], -1), torch.stack([shear_x, one], -1)], -2)
    rot_m = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    m = (rot_m @ shear_m) * scale[:, None, None]
    x = torch.einsum("bij,btpj->btpi", m, x - 0.5) + 0.5 + shift[:, None, None, None]

    # hide 20-40% of the sign's frames in one stretch (p 0.5)
    length = mask.sum(1).float()
    size = (length * (0.2 + 0.2 * torch.rand(b, device=dev))).long()
    start = (torch.rand(b, device=dev) * (length - size).clamp(min=1)).long()
    hide = chance(0.5)[:, None] & (j >= start[:, None]) & (j < (start + size)[:, None])
    x = torch.where(hide[:, :, None, None], torch.nan, x)

    # hide every point inside a random square of side 0.2-0.4 (p 0.5)
    corner = torch.rand(b, 2, device=dev)
    side = 0.2 + 0.2 * torch.rand(b, device=dev)
    inside = ((x > corner[:, None, None]) & (x < corner[:, None, None] + side[:, None, None, None])).all(-1)
    inside &= chance(0.5)[:, None, None]
    x = torch.where(inside[..., None], torch.nan, x)
    return x, mask


def features(x):
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


model = SignModelV3(N_POINTS * 6, N_CLASSES).to(device)
print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
try:
    opt = torch.optim.RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)
except TypeError:  # older PyTorch
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_idx) / BATCH)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps_per_epoch, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=device == "cuda")
params = [p for p in model.parameters() if p.requires_grad]


def evaluate(idx):
    model.eval()
    logits, labels = [], []
    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
        for x, mask, t in batches(idx, shuffle=False):
            logits.append(model(features(x), mask).float())
            labels.append(t)
    logits, labels = torch.cat(logits), torch.cat(labels)
    top5 = logits.topk(min(5, N_CLASSES), 1).indices
    return logits, labels, (top5[:, 0] == labels).float().mean().item(), (top5 == labels[:, None]).any(1).float().mean().item()


def loss_of(xa, ma, t):
    with torch.autocast(device, enabled=device == "cuda"):
        out = model(features(xa), ma)
        return F.cross_entropy(out.float(), t, label_smoothing=0.1), out


# ---- resume after a Colab disconnect ----
LAST = os.path.join(OUT, "islr_v3_last.pt")
start_epoch, best, best_epoch = 1, -1.0, 0
if os.path.exists(LAST) and os.environ.get("RESUME", "1") == "1":
    ck = torch.load(LAST, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    start_epoch, best, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
    print(f"resuming after epoch {ck['epoch']} (best validation so far {best:.1%} at epoch {best_epoch})")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    model.late_drop_on = epoch > LATE_START
    use_awp = epoch > LATE_START
    t0, total, correct, seen = time.time(), 0.0, 0, 0
    for x, mask, t in batches(train_idx, shuffle=True):
        xa, ma = augment(x, mask)
        loss, out = loss_of(xa, ma, t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if use_awp:
            # push each weight tensor 0.2 along its gradient, compute the update there, then move back
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
    logits, labels, top1, top5 = evaluate(val_idx)
    print(f"epoch {epoch:3d}  loss {total / seen:.3f}  train acc {correct / seen:.1%}  "
          f"val top-1 {top1:.1%}  top-5 {top5:.1%}  {time.time() - t0:.0f} s"
          + ("  (late dropout + AWP on)" if epoch == LATE_START + 1 else ""))
    if top1 > best:
        best, best_epoch = top1, epoch
        torch.save({"arch": "v3", "state_dict": model.state_dict(), "n_in": N_POINTS * 6, "n_classes": N_CLASSES,
                    "dim": DIM, "max_len": MAX_LEN, "keep": info["keep"], "signs": info["signs"],
                    "val_top1": top1, "epoch": epoch}, os.path.join(OUT, "islr_model_v3.pt"))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch}, LAST)
    if PATIENCE and epoch - best_epoch >= PATIENCE:
        print(f"validation has not improved for {PATIENCE} epochs: stopping")
        break

# ---- final test with the best epoch's weights; the test signers were never used to choose anything ----
model.load_state_dict(torch.load(os.path.join(OUT, "islr_model_v3.pt"), map_location=device, weights_only=False)["state_dict"])
val_logits, val_labels, _, _ = evaluate(val_idx)
test_logits, test_labels, test_top1, test_top5 = evaluate(test_idx)
np.savez(os.path.join(OUT, "islr_v3_logits.npz"), val_logits=val_logits.cpu().numpy(), val_labels=val_labels.cpu().numpy(),
         test_logits=test_logits.cpu().numpy(), test_labels=test_labels.cpu().numpy(),
         test_signer=meta["participant_id"][np.sort(test_idx)])
print(f"best validation top-1: {best:.1%} (epoch {best_epoch})")
print(f"TEST top-1: {test_top1:.1%}  top-5: {test_top5:.1%}  (3 signers never seen in training or validation)")
correct = (test_logits.argmax(1) == test_labels).cpu().numpy()
signer = meta["participant_id"][np.sort(test_idx)]
for s in np.unique(signer):
    print(f"  test signer {s}: top-1 {correct[signer == s].mean():.1%} ({(signer == s).sum()} signs)")
