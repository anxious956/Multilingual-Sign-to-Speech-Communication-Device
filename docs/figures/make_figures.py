import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/user/Multilingual-Sign-to-Speech-Communication-Device/docs/figures"
INK = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#2a78d6"
ACCENT_FILL = "#e3eefb"
GRAY_FILL = "#f1f1ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})


def box(ax, x, y, w, h, text, fill=GRAY_FILL, edge=MUTED, bold=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10,
                                 color=color, lw=1.1, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ---------- Figure 1: system block diagram ----------
# Even grid: every box is W wide, columns are STEP apart, so every
# horizontal arrow has the same length.
fig, ax = plt.subplots(figsize=(7.6, 3.3))
W, H, STEP = 1.8, 1.3, 2.15
col = lambda i: 0.95 + i * STEP
yt, yb = 4.4, 1.6
ax.set_xlim(0, col(7) + 0.95); ax.set_ylim(-0.35, 6.2); ax.axis("off")
HW = dict(fill="white", edge=INK)
FS = 7.2

def harrow(i, j, y):
    """Arrow between neighbouring columns i -> j on row y."""
    d = 1 if j > i else -1
    arrow(ax, (col(i) + d * W / 2, y), (col(j) - d * W / 2, y))

# sign -> speech (top row, left to right)
top = ["Camera", "Landmarks\n(MediaPipe)", "Sign\nrecognizer\n(top-5 +\nconfidence)",
       "Sentence\nbuilder\n(English)", None, "Translation\nEN→TR/ES\n(NLLB-200)",
       "Speech\nsynthesis\n(Piper)", "Speaker"]
for i, t in enumerate(top):
    if t is None:
        continue
    box(ax, col(i), yt, W, H, t, fs=FS, **(HW if i in (0, 7) else {}))
for i in range(7):
    harrow(i, i + 1, yt)

# touchscreen spans both rows, same width as other boxes
ts_h = (yt - yb) + H
box(ax, col(4), (yt + yb) / 2, W, ts_h,
    "Touchscreen\n(faces the\nDeaf user)\n\n• confirm\n   sentence\n• read\n   reply\n• choose\n   language",
    fill=ACCENT_FILL, edge=ACCENT, fs=6.7)

# reply path (bottom row, right to left)
box(ax, col(7), yb, W, H, "Micro-\nphone", fs=FS, **HW)
box(ax, col(6), yb, W, H, "Voice\ndetection\n(VAD)", fs=FS)
box(ax, col(5), yb, W, H, "Speech\nrecognition\n→ English\n(Whisper)", fs=FS)
for i in (7, 6, 5):
    harrow(i, i - 1, yb)

# people
gap = yt - yb - H            # vertical space between the two rows
mid = (yt + yb) / 2
ax.text(col(0), yt + H / 2 + 0.6, "Deaf user", ha="center", va="center", fontsize=8, fontweight="bold")
arrow(ax, (col(0), yt + H / 2 + 0.4), (col(0), yt + H / 2))
ax.text(col(7), mid, "Hearing\nperson", ha="center", va="center", fontsize=8, fontweight="bold")
arrow(ax, (col(7), yt - H / 2), (col(7), mid + 0.38))
arrow(ax, (col(7), mid - 0.38), (col(7), yb + H / 2))

# notes
arrow(ax, (col(6), yt - H / 2), (col(6), yb + H / 2), ls="--")
ax.text(col(6) - 0.12, mid, "mic muted\nwhile device\nspeaks", ha="right", va="center", fontsize=6.3, color=MUTED)
ax.text(col(5), yt + H / 2 + 0.2, "skipped for English", ha="center", fontsize=6.3, color=MUTED)

# legend
box(ax, 0.75, -0.05, 0.7, 0.42, "", **HW)
ax.text(1.25, -0.05, "Hardware", va="center", fontsize=7)
box(ax, 3.3, -0.05, 0.7, 0.42, "")
ax.text(3.8, -0.05, "Software on the Jetson Orin Nano (fully offline)", va="center", fontsize=7)
fig.savefig(f"{OUT}/fig1_block_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Figure 2: confidence-gated decision flow ----------
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.set_xlim(0, 14.8); ax.set_ylim(-0.2, 7.6); ax.axis("off")
W, H = 2.9, 1.0
box(ax, 1.5, 5.9, W, H, "Rest pose ends\none sign")
box(ax, 5.0, 5.9, W, H, "Recognizer:\ntop-5 signs\n+ confidence")
box(ax, 8.7, 5.9, W + 0.2, H, "Confidence\n≥ threshold?", fill=ACCENT_FILL, edge=ACCENT, bold=True)
box(ax, 12.6, 5.9, W, H, "Accept top sign")
box(ax, 8.7, 3.6, W + 0.6, H + 0.2, "Show top-5 candidates.\nUser picks one\nor re-signs")
box(ax, 12.6, 3.6, W, H, "Sentence\ncomplete?")
box(ax, 12.6, 1.3, W, H, "Show English\nsentence", )
box(ax, 8.7, 1.3, W + 0.2, H, "User confirms?", fill=ACCENT_FILL, edge=ACCENT, bold=True)
box(ax, 4.2, 1.3, W + 0.4, H, "Translate and speak\n(EN / TR / ES)", fill="white", edge=INK, bold=True)

arrow(ax, (1.5 + W / 2, 5.9), (5.0 - W / 2, 5.9))
arrow(ax, (5.0 + W / 2, 5.9), (8.7 - (W + 0.2) / 2, 5.9))
arrow(ax, (8.7 + (W + 0.2) / 2, 5.9), (12.6 - W / 2, 5.9)); ax.text(10.6, 6.1, "yes", fontsize=7.5, color=MUTED)
arrow(ax, (8.7, 5.9 - H / 2), (8.7, 3.6 + H / 2)); ax.text(8.85, 4.75, "no", fontsize=7.5, color=MUTED)
arrow(ax, (12.6, 5.9 - H / 2), (12.6, 3.6 + H / 2))
arrow(ax, (8.7 + (W + 0.2) / 2, 3.6), (12.6 - W / 2, 3.6))
arrow(ax, (12.6, 3.6 - H / 2), (12.6, 1.3 + H / 2)); ax.text(12.75, 2.45, "yes", fontsize=7.5, color=MUTED)
# "no" loop back to next sign
ax.plot([12.6 + W / 2, 14.6, 14.6, 1.5], [3.6, 3.6, 7.2, 7.2], ls="--", color=MUTED, lw=1.1)
arrow(ax, (1.5, 7.2), (1.5, 5.9 + H / 2), ls="--")
ax.text(8.0, 7.35, "no: sign the next word", fontsize=7.5, color=MUTED, ha="center")
arrow(ax, (12.6 - W / 2, 1.3), (8.7 + (W + 0.2) / 2, 1.3))
arrow(ax, (8.7 - (W + 0.2) / 2, 1.3), (4.2 + (W + 0.4) / 2, 1.3)); ax.text(6.6, 1.5, "yes", fontsize=7.5, color=MUTED)
arrow(ax, (8.7, 1.3 - H / 2), (8.7, 0.2), ls="--")
ax.text(8.85, 0.25, "no: edit or re-sign (nothing is spoken)", fontsize=7.5, color=MUTED, va="center")
fig.savefig(f"{OUT}/fig2_decision_flow.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Figure 3: hardware preferences (Tran et al. 2023) ----------
labels = ["Mobile phone app", "Stand-alone system\n(our device type)", "Wearable", "Implantable"]
vals = [91, 69, 38, 6]
colors = [ "#c9c8c3", ACCENT, "#c9c8c3", "#c9c8c3"]
fig, ax = plt.subplots(figsize=(6.0, 2.4))
bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.55)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v}%", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("Participants comfortable using it (n = 32)", color=MUTED, fontsize=8.5)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c3")
ax.tick_params(axis="y", length=0, labelsize=8.5)
ax.tick_params(axis="x", colors=MUTED, labelsize=8)
ax.xaxis.grid(True, color="#ecebe8", lw=0.8); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/fig3_hardware_preferences.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")
