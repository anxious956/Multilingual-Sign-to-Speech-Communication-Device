"""Side-view line drawing of the device in use at a service counter."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Arc, FancyBboxPatch

OUT = "/home/user/Multilingual-Sign-to-Speech-Communication-Device/docs/figures/fig_setup_sketch.png"
K = "#111111"
LW = 1.6
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9})

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.set_xlim(0, 16); ax.set_ylim(-1.4, 8.2); ax.set_aspect("equal"); ax.axis("off")


def person(x, facing, arms_up):
    """Simple standing figure; facing = +1 (right) or -1 (left)."""
    ax.add_patch(Circle((x, 6.55), 0.5, fill=False, ec=K, lw=LW))
    ax.plot([x, x], [6.05, 3.2], color=K, lw=LW)                  # torso
    ax.plot([x, x - 0.45], [3.2, 0.0], color=K, lw=LW)             # legs
    ax.plot([x, x + 0.45], [3.2, 0.0], color=K, lw=LW)
    if arms_up:                                                     # signing
        ax.plot([x, x + facing * 0.9, x + facing * 1.25], [5.5, 4.7, 5.55], color=K, lw=LW)
        ax.plot([x, x + facing * 0.7, x + facing * 1.05], [5.5, 4.5, 5.0], color=K, lw=LW)
        ax.add_patch(Circle((x + facing * 1.28, 5.62), 0.13, color=K))
        ax.add_patch(Circle((x + facing * 1.08, 5.05), 0.13, color=K))
    else:
        ax.plot([x, x + facing * 0.55, x + facing * 1.1], [5.5, 4.4, 3.85], color=K, lw=LW)
    # eye marks the facing direction
    ax.add_patch(Circle((x + facing * 0.22, 6.65), 0.06, color=K))


# floor and counter
ax.plot([0.2, 15.8], [0, 0], color=K, lw=LW)
ax.add_patch(Rectangle((5.2, 0), 5.6, 3.35, fill=False, ec=K, lw=LW, hatch="////"))
ax.add_patch(Rectangle((4.9, 3.35), 6.2, 0.25, fc="white", ec=K, lw=LW))

person(2.6, +1, arms_up=True)     # Deaf user signing
person(13.4, -1, arms_up=False)   # hearing staff member

# device: base (Jetson inside), stand, touchscreen tilted toward Deaf user
top = 3.6
ax.add_patch(Rectangle((6.6, top), 2.8, 0.55, fc="white", ec=K, lw=LW))            # base
ax.plot([8.0, 7.17], [top + 0.55, top + 1.53], color=K, lw=LW)                       # stand
screen = Polygon([[6.65, top + 1.0], [7.2, top + 3.1], [7.55, top + 3.0], [7.0, top + 0.9]],
                 closed=True, fc="white", ec=K, lw=LW)
ax.add_patch(screen)
ax.add_patch(Circle((7.33, top + 3.22), 0.12, color=K))                              # camera
# speaker and microphone on the staff side
ax.add_patch(FancyBboxPatch((9.0, top + 0.62), 0.55, 0.9, boxstyle="round,pad=0.02",
                            fc="white", ec=K, lw=LW))
for yy in (top + 0.82, top + 1.02, top + 1.22):
    ax.plot([9.12, 9.43], [yy, yy], color=K, lw=0.9)
ax.plot([9.9, 9.9], [top + 0.55, top + 1.55], color=K, lw=LW)
ax.add_patch(Circle((9.9, top + 1.65), 0.14, fill=False, ec=K, lw=LW))              # mic

# camera field of view toward the signer
for y_end in (7.2, 4.1):
    ax.plot([7.25, 3.9], [top + 3.2, y_end], color=K, lw=0.9, ls=(0, (4, 3)))

# sound: speaker -> staff, staff -> microphone
for r in (0.45, 0.8, 1.15):
    ax.add_patch(Arc((9.6, top + 1.05), r * 2, r * 2, theta1=-35, theta2=35, color=K, lw=1.0))
for r in (0.45, 0.8):
    ax.add_patch(Arc((12.6, 6.35), r * 2, r * 2, theta1=145, theta2=215, color=K, lw=1.0))

# numbered callouts, patent-drawing style
callouts = [
    (1, (7.33, top + 3.22), (6.1, 7.9)),
    (2, (6.95, top + 2.0), (5.55, 6.2)),
    (3, (8.0, top + 0.28), (8.0, 1.9)),
    (4, (9.27, top + 1.45), (10.2, 6.7)),
    (5, (9.9, top + 1.78), (11.1, 7.6)),
]
for n, (px, py), (tx, ty) in callouts:
    ax.plot([px, tx], [py, ty], color=K, lw=0.8)
    ax.add_patch(Circle((tx, ty), 0.32, fc="white", ec=K, lw=1.0, zorder=3))
    ax.text(tx, ty, str(n), ha="center", va="center", fontsize=8.5, zorder=4)

ax.text(2.6, -0.55, "Deaf user (signs in ASL)", ha="center", va="top", fontsize=8.5)
ax.text(13.4, -0.55, "Hearing staff member", ha="center", va="top", fontsize=8.5)
ax.text(8.0, -0.55, "Service counter", ha="center", va="top", fontsize=8.5)

legend = ("1  Camera      2  Touchscreen (confirm sentence, read reply)      "
          "3  Jetson Orin Nano (in base)      4  Speaker      5  Microphone")
ax.text(8.0, -1.25, legend, ha="center", va="center", fontsize=7.6)

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("ok")
