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
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.set_xlim(0, 16.6); ax.set_ylim(-0.3, 8.1); ax.axis("off")
w, h = 1.75, 1.15
yt, yb = 6.2, 2.2
HW = dict(fill="white", edge=INK)

# people
ax.text(1.0, 7.55, "Deaf user", ha="center", fontsize=8, fontweight="bold")
ax.text(15.6, 4.2, "Hearing\nperson", ha="center", va="center", fontsize=8, fontweight="bold")
arrow(ax, (1.0, 7.35), (1.0, yt + h / 2))                     # signs to camera
arrow(ax, (15.6, yt - h / 2), (15.6, 4.2 + 0.45))              # hears speaker
arrow(ax, (15.6, 4.2 - 0.45), (15.6, yb + h / 2))              # speaks into mic

# sign -> speech path (top, left to right)
box(ax, 1.0, yt, 1.4, h, "Camera", **HW)
top = [(3.0, "Landmarks\n(MediaPipe)"), (5.0, "Sign\nrecognizer\n(top-5 +\nconfidence)"),
       (7.0, "Sentence\nbuilder\n(English)")]
for x, t in top:
    box(ax, x, yt, w, h + 0.3, t)
arrow(ax, (1.7, yt), (3.0 - w / 2, yt))
for a, b in zip(top, top[1:]):
    arrow(ax, (a[0] + w / 2, yt), (b[0] - w / 2, yt))

# touchscreen (hardware, shared by both paths)
box(ax, 9.0, 4.2, 2.1, 6.1,
    "Touchscreen\n(faces the\nDeaf user)\n\n\u2022 confirm or\n   edit the\n   sentence\n\n\u2022 read the\n   reply\n\n\u2022 choose\n   output\n   language",
    fill=ACCENT_FILL, edge=ACCENT, fs=6.9)
arrow(ax, (7.0 + w / 2, yt), (9.0 - 1.05, yt))

right = [(11.6, "Translation\nEN\u2192TR/ES\n(NLLB-200)"), (13.6, "Speech\nsynthesis\n(Piper)")]
for x, t in right:
    box(ax, x, yt, w, h, t)
arrow(ax, (9.0 + 1.05, yt), (11.6 - w / 2, yt))
ax.text(11.6, yt - h / 2 - 0.3, "skipped for English", ha="center", fontsize=6.3, color=MUTED)
arrow(ax, (11.6 + w / 2, yt), (13.6 - w / 2, yt))
box(ax, 15.6, yt, 1.4, h, "Speaker", **HW)
arrow(ax, (13.6 + w / 2, yt), (15.6 - 0.7, yt))

# reply path (bottom, right to left)
box(ax, 15.6, yb, 1.4, h, "Micro-\nphone", **HW)
bot = [(13.6, "Voice\ndetection\n(VAD)"), (11.6, "Speech\nrecognition\n\u2192 English\n(Whisper)")]
for x, t in bot:
    box(ax, x, yb, w, h + 0.15, t)
arrow(ax, (15.6 - 0.7, yb), (13.6 + w / 2, yb))
arrow(ax, (13.6 - w / 2, yb), (11.6 + w / 2, yb))
arrow(ax, (11.6 - w / 2, yb), (9.0 + 1.05, yb))

# mic muted while the device speaks
arrow(ax, (13.6, yt - h / 2), (13.6, yb + (h + 0.15) / 2), ls="--")
ax.text(13.45, 4.2, "mic muted\nwhile device\nspeaks", fontsize=6.3, color=MUTED, va="center", ha="right")

# legend
box(ax, 1.0, 0.45, 0.9, 0.5, "", **HW)
ax.text(1.6, 0.45, "Hardware", va="center", fontsize=7)
box(ax, 4.0, 0.45, 0.9, 0.5, "")
ax.text(4.6, 0.45, "Software on the Jetson Orin Nano (fully offline)", va="center", fontsize=7)
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
