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
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.set_xlim(0, 15); ax.set_ylim(0, 6.8); ax.axis("off")

ax.add_patch(FancyBboxPatch((1.55, 0.35), 11.9, 6.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="none", ec=ACCENT, lw=1.2, ls="--"))
ax.text(7.5, 6.05, "On-device processing: NVIDIA Jetson Orin Nano (fully offline)",
        ha="center", va="center", fontsize=8.5, color=ACCENT, fontweight="bold")

# Sign -> speech row
y1, y2, w, h = 4.9, 3.0, 2.05, 1.05
box(ax, 0.7, y1, 1.3, h, "Camera", fill="white", edge=INK)
row1 = [(2.85, "Landmark\nextraction\n(MediaPipe)"), (5.15, "Sign\nrecognizer\n(top-k, conf.)"),
        (7.45, "Sentence\nbuilder\n(English)"), (9.75, "User\nconfirms\non screen"),
        (12.05, "Translation\nEN→TR/ES\n(NLLB-200)")]
for x, t in row1:
    box(ax, x, y1, w, h, t, fill=ACCENT_FILL if "confirms" in t else GRAY_FILL,
        edge=ACCENT if "confirms" in t else MUTED, bold="confirms" in t)
xs = [0.75 + 0.6] + [x for x, _ in row1]
arrow(ax, (1.35, y1), (2.85 - w / 2, y1))
for a, b in zip(row1, row1[1:]):
    arrow(ax, (a[0] + w / 2, y1), (b[0] - w / 2, y1))

# TTS + speaker below translation
box(ax, 12.05, y2, w, h, "Speech\nsynthesis\n(Piper TTS)")
arrow(ax, (12.05, y1 - h / 2), (12.05, y2 + h / 2))
box(ax, 14.3, y2, 1.3, h, "Speaker", fill="white", edge=INK)
arrow(ax, (12.05 + w / 2, y2), (14.3 - 0.65, y2))

# Reply row
y3 = 1.25
box(ax, 0.7, y3, 1.3, h, "Micro-\nphone", fill="white", edge=INK)
box(ax, 5.15, y3, w + 1.0, h, "Speech recognition\nTR/ES/EN → English\n(Whisper)")
box(ax, 9.75, y3, w + 0.6, h, "English text\non display")
arrow(ax, (1.35, y3), (5.15 - (w + 1.0) / 2, y3))
arrow(ax, (5.15 + (w + 1.0) / 2, y3), (9.75 - (w + 0.6) / 2, y3))
ax.text(0.7, y1 + 0.85, "Deaf user signs", ha="center", fontsize=7.5, color=MUTED)
ax.text(0.7, y3 - 0.85, "Hearing person\nreplies", ha="center", va="top", fontsize=7.5, color=MUTED)
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
