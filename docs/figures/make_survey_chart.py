"""Figure: mean survey ratings by setting, from Tran et al. (ASSETS 2023).

The response counts below were read from Figures 1 and 2 of the paper
(32 participants, 5-point scale). Every bar segment measured as an exact
multiple of 1/32, so the counts are exact. Each row lists how many
participants gave ratings 1, 2, 3, 4 and 5.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/Multilingual-Sign-to-Speech-Communication-Device/docs/figures/fig_survey_means.png"

# Figure 1 of the paper: how often participants wanted an interpreter but
# could not get one (1 = never, 5 = very often).
UNMET = {
    "Medical settings": (3, 7, 6, 11, 5),
    "Professional settings": (5, 3, 10, 8, 6),
    "Education settings": (4, 7, 8, 7, 6),
    "Formal personal events": (7, 7, 7, 4, 7),
    "Going into a business": (10, 6, 5, 2, 9),
    "Legal settings": (9, 7, 4, 6, 6),
    "Large group social interactions": (8, 6, 7, 7, 4),
    "Informal educational activities": (7, 7, 9, 6, 3),
    "Social media": (11, 7, 1, 6, 7),
    "Mental healthcare": (10, 5, 9, 4, 4),
    "Performing arts": (8, 10, 8, 4, 2),
    "One-on-one social interactions": (16, 4, 4, 5, 3),
    "Reading English text": (19, 4, 6, 1, 2),
    "Self-service kiosks": (21, 2, 5, 1, 3),
}
# Figure 2 of the paper: willingness to use automatic sign language
# translation (1 = not at all likely, 5 = very likely).
WILLING = {
    "Going into a business": (0, 5, 9, 8, 10),
    "Self-service kiosks": (3, 4, 6, 7, 12),
    "Informal educational activities": (3, 4, 8, 6, 11),
    "Social media": (5, 2, 8, 6, 11),
    "Medical settings": (10, 1, 3, 5, 13),
    "Professional settings": (8, 1, 8, 4, 11),
    "Large group social interactions": (7, 2, 7, 7, 9),
    "One-on-one social interactions": (5, 7, 6, 5, 9),
    "Performing arts": (6, 4, 9, 6, 7),
    "Legal settings": (9, 3, 7, 4, 9),
    "Reading English text": (8, 5, 7, 3, 9),
    "Mental healthcare": (12, 1, 5, 4, 10),
    "Education settings": (10, 4, 5, 4, 9),
    "Formal personal events": (10, 4, 4, 6, 8),
}


def mean(counts):
    assert sum(counts) == 32
    return sum((i + 1) * c for i, c in enumerate(counts)) / 32


settings = sorted(WILLING, key=lambda s: mean(WILLING[s]))  # bottom -> top
TARGET = "Going into a business"
INK, MUTED, BAR, HIGHLIGHT, GRID = "#111111", "#555555", "#bdbdbd", "#3a3a3a", "#e6e6e6"

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True,
                         gridspec_kw={"wspace": 0.08})
panels = [
    (UNMET, "(a) Wanted an interpreter\nbut could not get one", "1 = never,  5 = very often"),
    (WILLING, "(b) Willing to use automatic\nsign language translation", "1 = not at all likely,  5 = very likely"),
]
y = range(len(settings))
for ax, (data, title, scale) in zip(axes, panels):
    vals = [mean(data[s]) for s in settings]
    colors = [HIGHLIGHT if s == TARGET else BAR for s in settings]
    ax.barh(list(y), [v - 1 for v in vals], left=1, height=0.62, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 0.04, yi, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlabel(f"Mean rating ({scale})", fontsize=8.5, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)

axes[0].set_yticks(list(y))
axes[0].set_yticklabels(settings)
for lbl in axes[0].get_yticklabels():
    if lbl.get_text() == TARGET:
        lbl.set_fontweight("bold")

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("ok")
