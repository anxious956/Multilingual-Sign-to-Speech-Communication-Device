"""Figures for proposal Section 7 (design methodology, prototyping, validation, hardware block diagram).

python docs/figures/make_section7_figures.py [flow|block|results|sim|vmodel ...]   (default: all)

Numbers come from docs/GOAL1_LOG.md, docs/figures/calib_v5_report.md and docs/figures/section7_benchmarks.json
(written by tools/bench_section7.py). Same style as the other proposal figures: Liberation Serif, black and greys.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = "#111111"          # lines and text
G1 = "#3a3a3a"         # dark fill (highlight, as in Fig. 1)
G2 = "#bdbdbd"         # mid grey
G3 = "#e6e6e6"         # light fill
G4 = "#f4f4f4"         # very light fill
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.edgecolor": "#555555",
                     "axes.linewidth": 0.8, "xtick.color": "#333333", "ytick.color": "#333333"})


PT = 1 / 36  # data units per typographic point; each figure sets it (axes fill the figure)


def setup(w_in, h_in, w_units, h_units):
    """Figure whose axes fill the whole canvas, so that 1 data unit = w_in / w_units inches."""
    global PT
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_units); ax.set_ylim(0, h_units); ax.axis("off")
    PT = h_units / (h_in * 72)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), status=None, rounded=False, fc="white", lw=1.3, title_size=10.5,
        line_size=8.3, title_weight="bold", align="center", ec=K, title_color=K):
    """Box with a bold title and smaller detail lines; (x, y) is the lower-left corner."""
    style = "round,pad=0,rounding_size=0.18" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    tx = x + w / 2 if align == "center" else x + 8 * PT
    yy = y + h - 6 * PT
    for t in title.split("\n"):
        ax.text(tx, yy, t, ha=align, va="top", fontsize=title_size, fontweight=title_weight, zorder=3,
                color=title_color)
        yy -= title_size * 1.18 * PT
    yy -= 3 * PT
    for ln in lines:
        ax.text(tx, yy, ln, ha=align, va="top", fontsize=line_size, zorder=3, color="#222222")
        yy -= line_size * 1.22 * PT
    if status:
        ax.text(x + w / 2, y + 5 * PT, status, ha="center", va="bottom", fontsize=line_size - 0.3, style="italic",
                zorder=3, color="#222222")


def box_height(title, lines=(), status=False, title_size=10.5, line_size=8.3):
    n_t = title.count("\n") + 1
    return (5 + n_t * title_size * 1.18 + 3 + len(lines) * line_size * 1.22 + (line_size * 1.5 if status else 0)) * PT


def arrow(ax, x0, y0, x1, y1, lw=1.2, ls="-", head=8, color=K, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head / 20},head_width={head / 40}",
                                 mutation_scale=10, lw=lw, ls=ls, color=color, zorder=4, connectionstyle=cs,
                                 shrinkA=0, shrinkB=0))


def double_v(ax, x, y_top, y_bot, gap=0.09):
    """Down arrow and up arrow side by side, as in the course template."""
    arrow(ax, x - gap, y_top, x - gap, y_bot)
    arrow(ax, x + gap, y_bot, x + gap, y_top)


def double_h(ax, x_left, x_right, y, gap=0.09):
    arrow(ax, x_left, y + gap, x_right, y + gap)
    arrow(ax, x_right, y - gap, x_left, y - gap)


# ---------------------------------------------------------------- Fig. 5: design flow chart
def fig_flow():
    """The course's recommended design flow chart, filled in for this project."""
    global PT
    W = 13.7
    PT = 1 / (0.5 * 72)  # 1 unit = 0.5 in
    req = ("Design Objectives, Requirements, Specifications",
           ["≥ 75% top-1 and ≥ 90% top-5 accuracy on signers not seen in training",
            "100–250 service-counter signs, chosen with Deaf ASL users",
            "< 1.2 s from confirmation to speech; English, Turkish, Spanish",
            "Fully offline; ≥ 15 frames/s; all models within 8 GB (Tables 1, 3)"])
    bench = ("Benchmarking", ["ASL Citizen baseline: 63.1% [2]", "Kaggle 1st-place model [7]",
                              "Patented systems (Table 2)", "Jetson specifications [30]"])
    concept = ("Design Concept and Constraints",
               ["Single RGB camera → body and hand landmarks → isolated-sign",
                "classifier with calibrated confidence → abstain or confirm",
                "on screen → offline translation and speech (Table 10)",
                "Constraints: memory, timing, licences, safety (Section 6.2)"])
    biz = ("Business and Economic\nPerspectives", ["≈ $485–640 in parts (Table 6)", "No cloud or per-use fees",
                                                    "Non-commercial licences"])
    comps = [("Component X Design:\nSign Recognition (Goal 1)",
              ["MediaPipe landmarks (94 points)", "Rest-pose sign segmentation", "1D-CNN + transformer classifier",
               "Temperature scaling and threshold"], "Laptop prototype: 86.6% top-1"),
             ("Component Y Design:\nLanguage and Speech (Goal 2)",
              ["Templates + small language model", "Touchscreen confirmation", "NLLB-200 translation, Piper TTS",
               "Whisper reply channel"], "Models benchmarked on a CPU"),
             ("Component Z Design:\nEmbedded Platform (Goal 3)",
              ["Jetson Orin Nano, 8 GB memory", "ONNX Runtime / TensorRT", "Camera, touchscreen, mic, speaker",
               "Power, heat, enclosure"], "Parts selected; purchase next")]
    proto = ("Prototyping, Testing and Integration",
             ["Prototypes P1–P5 (Table 14): laptop demonstrator → interface mock-up → Jetson bench → enclosure",
              "Unit → integration → system tests against Table 16; components join the Jetson one at a time",
              "Validation with Deaf ASL signers on the device's own camera (Section 7.7)"])
    asm = ("Prototype Assembly", ["P5: integrated device in a 3D-printed enclosure (Fig. 6)",
                                  "Acceptance demonstration at a mock service counter",
                                  "Results reported against Tables 3 and 16 in ECE 416"])
    ts, ls = 10.5, 8.4
    h1 = max(box_height(req[0], req[1], False, ts, ls), box_height(bench[0], bench[1], False, ts, ls)) + 0.1
    h2 = max(box_height(concept[0], concept[1], False, ts, ls), box_height(biz[0], biz[1], False, ts, ls)) + 0.1
    h3 = box_height(comps[0][0], comps[0][1], True, 9.6, ls) + 0.1
    h4 = box_height(proto[0], proto[1], False, ts, ls) + 0.1
    h5 = box_height(asm[0], asm[1], False, ts, ls) + 0.1
    gap, gap_c = 0.62, 0.95
    H = 0.1 + h5 + gap + h4 + gap + h3 + gap_c + h2 + gap + h1 + 0.1
    fig, ax = setup(W * 0.5, H * 0.5, W, H)
    y5 = 0.1; y4 = y5 + h5 + gap; y3 = y4 + h4 + gap; y2 = y3 + h3 + gap_c; y1 = y2 + h2 + gap
    L, R = 1.05, 9.0           # main column
    SL, SR = 9.7, 13.55        # side column

    box(ax, L, y1, R - L, h1, req[0], req[1], title_size=ts, line_size=ls)
    box(ax, SL, y1, SR - SL, h1, bench[0], bench[1], rounded=True, title_size=ts, line_size=ls, fc=G4)
    double_h(ax, R, SL, y1 + h1 / 2)
    double_v(ax, (L + R) / 2, y1, y2 + h2)

    box(ax, L, y2, R - L, h2, concept[0], concept[1], title_size=ts, line_size=ls)
    box(ax, SL, y2, SR - SL, h2, biz[0], biz[1], rounded=True, title_size=ts, line_size=ls, fc=G4)
    double_h(ax, R, SL, y2 + h2 / 2)

    cw = 4.15
    xs = [0.62, 5.01, 9.4]
    for x, (t, l, st) in zip(xs, comps):
        box(ax, x, y3, cw, h3, t, l, status=st, title_size=9.6, line_size=ls)
    top3 = y3 + h3
    for x in xs[:2]:
        cx = x + cw / 2
        double_v(ax, max(cx, L + 0.6), y2, top3)
    # right-hand component sits under the side column: join it to the concept box with an elbow
    cx = xs[2] + cw / 2
    ym = (y2 + top3) / 2
    for xc, xb, down in ((cx - 0.09, R - 0.42, True), (cx + 0.09, R - 0.24, False)):
        yl = ym + (-0.09 if down else 0.09)
        if down:
            ax.plot([xb, xb, xc], [y2, yl, yl], color=K, lw=1.2, zorder=4, solid_joinstyle="miter")
            arrow(ax, xc, yl, xc, top3)
        else:
            ax.plot([xc, xc, xb], [top3, yl, yl], color=K, lw=1.2, zorder=4, solid_joinstyle="miter")
            arrow(ax, xb, yl, xb, y2)

    box(ax, L, y4, SR - L - 0.6, h4, proto[0], proto[1], title_size=ts, line_size=ls)
    for x in xs:
        arrow(ax, min(x + cw / 2, SR - 0.9), y3, min(x + cw / 2, SR - 0.9), y4 + h4)

    aw = 8.6
    ax0 = (L + SR - 0.6) / 2 - aw / 2
    box(ax, ax0, y5, aw, h5, asm[0], asm[1], title_size=ts, line_size=ls)
    arrow(ax, ax0 + aw / 2, y4, ax0 + aw / 2, y5 + h5)

    # feedback loop (added to the template): a failed test sends the work back to the concept
    xl = 0.36
    dash = (0, (4, 2.5))
    ax.plot([L, xl, xl], [y4 + h4 * 0.35, y4 + h4 * 0.35, y2 + h2 * 0.3], color=K, lw=1.0, ls=dash, zorder=1)
    arrow(ax, xl, y2 + h2 * 0.3, L, y2 + h2 * 0.3, lw=1.0, ls=dash)
    ax.text(xl - 0.1, (y4 + h4 * 0.35 + y2 + h2 * 0.3) / 2, "redesign when a test misses a target", rotation=90,
            ha="right", va="center", fontsize=7.8, style="italic", color="#222222")
    fig.savefig(os.path.join(HERE, "fig_design_flow.png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6: hardware block diagram
def line(ax, pts, lw=1.2, ls="-", color=K, z=4):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_joinstyle="miter", solid_capstyle="butt")


def label(ax, x, y, t, size=7.4, ha="center", va="bottom", style="normal", rot=0):
    ax.text(x, y, t, ha=ha, va=va, fontsize=size, style=style, color="#222222", zorder=6, rotation=rot,
            bbox=dict(fc="white", ec="none", pad=0.6))


def fig_block():
    W, H = 13.6, 12.9
    fig, ax = setup(6.8, 6.45, W, H)
    ts, ls = 9.2, 7.8
    PW = (0, (5, 2))  # power line pattern
    PL, PR = 2.75, 10.85          # right edge of the left peripherals, left edge of the right peripherals
    jx0, jx1, jy0, jy1 = 4.05, 9.55, 3.2, 12.0

    for x, t in ((PL / 2, "Faces the Deaf user"), ((jx0 + jx1) / 2, "Inside the base"),
                 ((PR + W) / 2, "Faces the hearing person")):
        ax.text(x, H - 0.12, t, ha="center", va="top", fontsize=9.4, style="italic", color="#333333")

    # ---- peripherals, Deaf user side
    box(ax, 0.05, 9.7, PL - 0.05, 1.85, "RGB camera", ["1080p, 30 frames/s", "USB video class", "Captures the signer"],
        title_size=ts, line_size=ls)
    box(ax, 0.05, 6.35, PL - 0.05, 2.6, "7-inch touchscreen",
        ["Video in: HDMI", "Touch out: USB", "Confirm the sentence,", "pick candidates,", "read the reply"],
        title_size=ts, line_size=ls)
    box(ax, 0.05, 4.0, PL - 0.05, 1.85, "Button and LED",
        ["Backup sign boundary;", "LED lit while camera", "or microphone is on"], title_size=ts, line_size=ls)

    # ---- Jetson developer kit (carrier board) with the module
    ax.add_patch(Rectangle((jx0, jy0), jx1 - jx0, jy1 - jy0, fc=G4, ec=K, lw=1.6, zorder=1))
    ax.text((jx0 + jx1) / 2, jy1 - 0.1, "NVIDIA Jetson Orin Nano Developer Kit", ha="center", va="top",
            fontsize=9.6, fontweight="bold")
    ax.text((jx0 + jx1) / 2, jy1 - 0.47, "carrier board", ha="center", va="top", fontsize=8, style="italic")
    mx0, mx1, my0, my1 = jx0 + 0.25, jx1 - 0.25, 8.25, 11.2
    ax.add_patch(Rectangle((mx0, my0), mx1 - mx0, my1 - my0, fc=G1, ec=K, lw=1.3, zorder=2))
    ax.text((mx0 + mx1) / 2, my1 - 0.1, "Jetson Orin Nano 8 GB module", ha="center", va="top", fontsize=9.4,
            fontweight="bold", color="white", zorder=3)
    sub = [("CPU", "6-core Arm Cortex-A78AE"), ("GPU", "Ampere, 1024 CUDA cores"),
           ("Memory", "8 GB LPDDR5, CPU and GPU")]
    for i, (a_, t) in enumerate(sub):
        yy = my1 - 0.6 - i * 0.6
        ax.add_patch(Rectangle((mx0 + 0.18, yy - 0.48), mx1 - mx0 - 0.36, 0.48, fc="white", ec=K, lw=0.9, zorder=3))
        ax.text(mx0 + 0.32, yy - 0.24, a_, ha="left", va="center", fontsize=8.2, fontweight="bold", zorder=4)
        ax.text(mx0 + 1.45, yy - 0.24, t, ha="left", va="center", fontsize=8.0, zorder=4)
    ax.text((mx0 + mx1) / 2, my0 + 0.1, "runs every model; cooled by heatsink and fan", ha="center",
            va="bottom", fontsize=7.4, style="italic", color="white", zorder=4)

    def iface(x, y, w, t, sub=None, dotted=False, h=0.82):
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=K, lw=1.0, ls=(0, (1.5, 1.5)) if dotted else "-", zorder=3))
        c = "#555555" if dotted else K
        ax.text(x + w / 2, y + h - 0.09, t, ha="center", va="top", fontsize=8.0, fontweight="bold", zorder=4, color=c)
        if sub:
            ax.text(x + w / 2, y + 0.08, sub, ha="center", va="bottom", fontsize=7.2, zorder=4, color=c)
    ix0, ix1 = jx0 + 0.25, jx1 - 0.25
    iw = (ix1 - ix0 - 0.2) / 2
    usb_y = 7.0
    iface(ix0, usb_y, ix1 - ix0, "USB 3.2 ports (4 × Type-A)", "data, and 5 V power for the peripherals")
    rows = [(5.8, ("DisplayPort", "only video output"), ("M.2 Key-M slot", "PCIe 3.0 ×4")),
            (4.6, ("40-pin header", "GPIO pins"), ("M.2 Key-E slot", "Wi-Fi card removed")),
            (3.4, ("DC power jack", "5.5 × 2.5 mm"), ("Ethernet", "unplugged in use"))]
    for y, (t1, s1), (t2, s2) in rows:
        iface(ix0, y, iw, t1, s1)
        iface(ix0 + iw + 0.2, y, iw, t2, s2, dotted=t2 in ("M.2 Key-E slot", "Ethernet"))
    line(ax, [((jx0 + jx1) / 2, my0), ((jx0 + jx1) / 2, usb_y + 0.82)], lw=2.6)
    label(ax, (jx0 + jx1) / 2 + 0.12, (my0 + usb_y + 0.82) / 2, "module connector", size=7, ha="left", va="center")

    # ---- peripherals, hearing person side
    box(ax, PR, 9.7, W - PR - 0.05, 1.85, "Powered speaker", ["USB audio", "Speaks the confirmed", "sentence"],
        title_size=ts, line_size=ls)
    box(ax, PR, 6.35, W - PR - 0.05, 2.6, "Microphone", ["USB, directional", "Records the reply;", "muted while the",
                                                        "device speaks"], title_size=ts, line_size=ls)
    box(ax, PR, 4.0, W - PR - 0.05, 1.85, "NVMe SSD, 256 GB", ["OS, models, voices;", "no video or audio", "is stored"],
        title_size=ts, line_size=ls)

    # ---- data lines (arrow = direction of data); interface names sit on the lines
    uy = usb_y + 0.41
    xa = 3.4
    line(ax, [(PL, 10.6), (xa, 10.6), (xa, uy + 0.16)]); arrow(ax, xa, uy + 0.16, ix0, uy + 0.16)
    label(ax, xa, 9.3, "USB", va="center", size=7.2)
    arrow(ax, PL, uy - 0.16, ix0, uy - 0.16)
    label(ax, (PL + ix0) / 2 - 0.05, uy - 0.2, "USB (touch)", va="top", size=7.2)
    line(ax, [(ix0, 6.2), (xa, 6.2), (xa, 6.65)]); arrow(ax, xa, 6.65, PL, 6.65)
    label(ax, (PL + ix0) / 2 - 0.1, 6.1, "DP → HDMI", va="top", size=7.2)
    arrow(ax, PL, 5.15, ix0, 5.15); arrow(ax, ix0, 4.8, PL, 4.8)
    label(ax, (PL + ix0) / 2 + 0.2, 5.22, "GPIO", size=7.2)
    xb = PR - 0.65
    line(ax, [(ix1, uy + 0.16), (xb, uy + 0.16), (xb, 10.6)]); arrow(ax, xb, 10.6, PR, 10.6)
    label(ax, xb, 9.3, "USB", va="center", size=7.2)
    arrow(ax, PR, uy - 0.16, ix1, uy - 0.16)
    label(ax, (ix1 + PR) / 2 - 0.2, uy - 0.2, "USB", va="top", size=7.2)
    ssd = [(ix1, 6.2), (xb, 6.2), (xb, 4.95), (PR, 4.95)]
    line(ax, ssd[:-1]); arrow(ax, xb, 4.95, PR, 4.95); arrow(ax, xb + 0.01, 6.2, ix1, 6.2)
    label(ax, xb, 5.55, "PCIe", va="center", size=7.2)

    # ---- power
    pw_y = 1.15
    box(ax, 0.05, 0.55, 3.55, 1.25, "19 V DC wall adapter", ["counter or bench use"], title_size=ts, line_size=ls)
    box(ax, W - 3.6, 0.55, 3.55, 1.25, "Battery pack (optional)", ["19 V output, ≈ 100 Wh"], title_size=ts,
        line_size=ls)
    jack_x = ix0 + iw / 2
    box(ax, jack_x - 1.35, 0.55, 2.7, 1.25, "Power switch", ["selects the source"], title_size=ts, line_size=ls)
    line(ax, [(3.6, pw_y), (jack_x - 1.35, pw_y)], lw=1.8, ls=PW)
    line(ax, [(W - 3.6, pw_y), (jack_x + 1.35, pw_y)], lw=1.8, ls=PW)
    line(ax, [(jack_x, 1.8), (jack_x, 3.25)], lw=1.8, ls=PW)
    arrow(ax, jack_x, 3.2, jack_x, 3.4, lw=1.8)
    label(ax, jack_x + 0.12, 2.5, "power", ha="left", va="center", size=7.2, style="italic")

    # legend
    lx, ly = jack_x + 2.0, 2.65
    arrow(ax, lx, ly, lx + 0.6, ly)
    ax.text(lx + 0.72, ly, "data (arrow = direction)", va="center", fontsize=7.4)
    line(ax, [(lx, ly - 0.42), (lx + 0.6, ly - 0.42)], lw=1.8, ls=PW)
    ax.text(lx + 0.72, ly - 0.42, "power", va="center", fontsize=7.4)
    ax.add_patch(Rectangle((lx + 1.95, ly - 0.54), 0.45, 0.24, fc="white", ec=K, lw=1.0, ls=(0, (1.5, 1.5))))
    ax.text(lx + 2.52, ly - 0.42, "not used (offline)", va="center", fontsize=7.4)

    fig.savefig(os.path.join(HERE, "fig_hw_block_diagram.png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 7: preliminary recognition results
# docs/GOAL1_LOG.md and docs/figures/calib_v5_report.md; test signers were never seen in training
VERSIONS = [("v1", 61.0, 77.6), ("v2", 62.3, 79.5), ("v3", 66.4, 83.0), ("v4", 67.2, 83.9)]   # Kaggle, 250 signs
V5 = ("v5", 86.6, 98.2)                                                                       # ASL Citizen, 134 words
V4_SIGNERS = [46.2, 71.1, 78.7]
V5_SIGNERS = [79.3, 87.8, 88.0, 83.4, 88.4, 84.3, 88.6, 88.9, 88.9, 87.1, 88.1]
V5_SIGNERS_TOP5 = [96.6, 97.9, 98.4, 97.5, 98.8, 97.0, 98.9, 98.8, 99.4, 99.4, 97.6]


def fig_results():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 3.15), gridspec_kw=dict(width_ratios=[1.4, 1], wspace=0.32))
    xs = [0, 1, 2, 3, 4.6]
    rows = VERSIONS + [V5]
    w = 0.36
    for x, (n, t1, t5) in zip(xs, rows):
        dark = n == "v5"
        a1.bar(x - w / 2, t1, w, color=G1 if dark else "#8c8c8c", zorder=2)
        a1.bar(x + w / 2, t5, w, color="#9a9a9a" if dark else G2, zorder=2)
        a1.text(x - w / 2, t1 + 0.8, f"{t1:.1f}", ha="center", va="bottom", fontsize=6.8, rotation=90,
                bbox=dict(fc="white", ec="none", pad=0.3))
        a1.text(x + w / 2, t5 + 0.8, f"{t5:.1f}", ha="center", va="bottom", fontsize=6.8, rotation=90,
                bbox=dict(fc="white", ec="none", pad=0.3))
    for yv in (75, 90):
        a1.axhline(yv, color=K, lw=0.9, ls=(0, (4, 2)), zorder=1)
    a1.axvline(3.8, color="#777777", lw=0.7)
    a1.set_xticks(xs, [r[0] for r in rows])
    a1.set_ylim(40, 112); a1.set_xlim(-0.6, 5.1)
    a1.set_ylabel("Accuracy on unseen signers (%)", fontsize=8.5)
    a1.tick_params(labelsize=8)
    a1.text(1.5, -0.12, "Kaggle ISLR: 250 signs, 3 test signers", transform=a1.get_xaxis_transform(), ha="center",
            va="top", fontsize=7.4, style="italic")
    a1.text(4.6, -0.12, "ASL Citizen: 134 words,\n11 Deaf test signers", transform=a1.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.4, style="italic")
    a1.bar([0], [0], color="#8c8c8c", label="top-1"); a1.bar([0], [0], color=G2, label="top-5")
    a1.plot([], [], color=K, lw=0.9, ls=(0, (4, 2)), label="targets: 75% top-1, 90% top-5")
    a1.legend(loc="upper left", fontsize=7.2, frameon=False, ncol=1, bbox_to_anchor=(-0.02, 1.02), handlelength=1.6,
              columnspacing=1.0)
    a1.set_title("(a) Recognizer versions", fontsize=9, loc="left")
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)

    rng = np.random.default_rng(3)
    for i, vals in enumerate((V4_SIGNERS, V5_SIGNERS)):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        a2.scatter(np.full(len(vals), i) + jitter, vals, s=22, color=G1 if i else "#8c8c8c", zorder=3, edgecolor="white",
                   linewidth=0.5)
        a2.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=K, lw=1.4, zorder=4)
        a2.text(i + 0.29, np.mean(vals), f"mean {np.mean(vals):.1f}", va="center", fontsize=7.2)
        a2.text(i + 0.29, min(vals), f"worst {min(vals):.1f}", va="center", fontsize=7.2)
    a2.axhline(75, color=K, lw=0.9, ls=(0, (4, 2)))
    a2.text(-0.42, 75.8, "75% target", ha="left", va="bottom", fontsize=7)
    a2.set_xticks([0, 1], ["v4, Kaggle\n(3 signers)", "v5, ASL Citizen\n(11 signers)"])
    a2.set_xlim(-0.45, 1.85); a2.set_ylim(40, 100)
    a2.set_ylabel("Top-1 accuracy per test signer (%)", fontsize=8.5)
    a2.tick_params(labelsize=8)
    a2.set_title("(b) Differences between signers", fontsize=9, loc="left")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    fig.savefig(os.path.join(HERE, "fig_prelim_results.png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # Fig. 8: the two v5 calibration plots from src/calibrate.py, side by side
    from PIL import Image
    ims = [Image.open(os.path.join(HERE, f"calib_v5_all_134_signs_{n}.png")).convert("RGB")
           for n in ("reliability", "risk_coverage")]
    h = max(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h)) for i in ims]
    out = Image.new("RGB", (sum(i.width for i in ims) + 40, h + 60), "white")
    x = 0
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(out)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 34)
    for tag, im in zip("ab", ims):
        out.paste(im, (x, 60)); d.text((x + 10, 8), f"({tag})", fill="black", font=font); x += im.width + 40
    out.save(os.path.join(HERE, "fig_calib_v5.png"))


# ---------------------------------------------------------------- Fig. 9: simulations and budgets
def fig_sim():
    d = json.load(open(os.path.join(HERE, "section7_benchmarks.json")))
    sm = d["simulation"]
    fig = plt.figure(figsize=(6.7, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.5], hspace=0.55, wspace=0.34)
    a1, a3, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    n = sm["n"]
    a1.plot(n, sm["corr_avg"], "-o", color=G1, ms=4, lw=1.5, label="mean of 11 test signers")
    a1.plot(n, sm["corr_worst"], "--s", color="#7a7a7a", ms=4, lw=1.3, label="weakest test signer")
    for x, y in zip(n, sm["corr_avg"]):
        if x in (1, 3, 6):
            a1.text(x + 0.12, y - 0.07, f"{y:.1f}", ha="left", va="top", fontsize=7)
    for x, y in zip(n, sm["corr_worst"]):
        if x in (1, 3, 6):
            a1.text(x, y + 0.1, f"{y:.1f}", ha="center", va="bottom", fontsize=7, color="#555555")
    a1.set_xlabel("Signs per sentence", fontsize=8.5); a1.set_ylabel("Expected corrections per sentence", fontsize=8.5)
    a1.set_xticks(n); a1.set_ylim(0, 2.2); a1.tick_params(labelsize=8)
    a1.legend(fontsize=7.2, frameon=False, loc="upper left")
    a1.set_title("(a) Corrections on the confirmation screen", fontsize=9, loc="left")

    langs = [("English", "en"), ("Turkish", "tr"), ("Spanish", "es")]
    for i, (name, k) in enumerate(langs):
        tr_s, tts_s = sm["lat_parts"][k]
        a3.barh(i, tr_s, color=G1, height=0.55)
        a3.barh(i, tts_s, left=tr_s, color=G2, height=0.55)
        a3.text(tr_s + tts_s + 0.03, i, f"{tr_s + tts_s:.2f} s", va="center", fontsize=7.4)
    a3.axvline(1.2, color=K, lw=1.0, ls=(0, (4, 2)))
    a3.text(1.18, 2.48, "1.2 s budget", ha="right", va="bottom", fontsize=7.2)
    a3.set_yticks(range(3), [l[0] for l in langs]); a3.invert_yaxis()
    a3.set_xlim(0, 1.45); a3.set_ylim(2.75, -0.55); a3.tick_params(labelsize=8)
    a3.set_xlabel("Seconds from confirmation to first audio", fontsize=8.5)
    a3.barh([0], [0], color=G1, label="translation (NLLB-200)")
    a3.barh([0], [0], color=G2, label="first audio (Piper)")
    a3.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.2, 1.03))
    a3.set_title("(b) Speech latency on a cloud CPU", fontsize=9, loc="left")

    mem = sm["mem"]
    order = [("Operating system and desktop", "OS and desktop*"), ("GPU runtime (CUDA, TensorRT)", "GPU runtime*"),
             ("Landmark extractor (MediaPipe)", "Landmarks*"), ("Sign recognizer (v5)", "Sign recognizer"),
             ("Sentence-builder language model", "Sentence model*"), ("NLLB-200, 8-bit", "NLLB-200, 8-bit"),
             ("Piper voices (3)", "Piper, 3 voices"), ("Whisper small", "Whisper small")]
    shades = ["#5a5a5a", "#8a8a8a", "#b0b0b0", "#000000", "#d2d2d2", "#2e2e2e", "#a0a0a0", "#6e6e6e"]
    x = 0
    for (k, lab), c in zip(order, shades):
        v = mem[k]
        assumed = lab.endswith("*")
        val = f"{v:.3f}" if v < 0.05 else f"{v:.2f}" if v < 1 else f"{v:.1f}"
        a2.barh(0, v, left=x, color=c, height=0.55, edgecolor="white", linewidth=0.8, hatch="////" if assumed else None,
                label=f"{lab} {val} GB")
        if v >= 0.45:
            dark = c in ("#5a5a5a", "#000000", "#2e2e2e", "#6e6e6e", "#8a8a8a")
            a2.text(x + v / 2, 0, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if dark else K,
                    bbox=dict(fc=c, ec="none", pad=0.4))
        x += v
    a2.axvline(8, color=K, lw=1.2)
    a2.text(7.93, 0.3, "8 GB of shared memory", ha="right", va="bottom", fontsize=7.4)
    a2.text(x + 0.08, 0, f"total {x:.1f} GB  (≈ {8 - x:.1f} GB left for working memory)", ha="left", va="center",
            fontsize=7.6)
    a2.set_xlim(0, 8.2); a2.set_ylim(-0.45, 0.6); a2.set_yticks([])
    a2.set_xlabel("Memory (GB)", fontsize=8.5)
    a2.tick_params(labelsize=8)
    a2.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.62), handlelength=1.4,
              columnspacing=1.2, title="* assumed allowance; all others measured from the model files",
              title_fontsize=7)
    a2.set_title("(c) Memory budget on the Jetson Orin Nano", fontsize=9, loc="left")
    for ax in (a1, a2, a3):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a2.spines["left"].set_visible(False)
    fig.savefig(os.path.join(HERE, "fig_simulations.png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 10: V-model with the prototypes
def fig_vmodel():
    W, H = 13.4, 8.6
    fig, ax = setup(6.7, 4.3, W, H)
    ts, ls = 8.8, 7.5
    bw, bh = 3.3, 1.12
    left = [("Stakeholder needs", "Deaf ASL users, service staff"),
            ("System requirements", "Tables 1 and 3"),
            ("Architecture", "Figs. 5 and 6, Table 12"),
            ("Component design", "Goals 1, 2 and 3")]
    right = [("Validation", "Deaf signers, mock counter"),
             ("System verification", "Table 16, on the Jetson"),
             ("Integration tests", "one model at a time"),
             ("Unit tests", "each component alone")]
    protos = ["P2 interface mock-up (early)\nP5 integrated device",
              "P5 integrated device",
              "P3 Jetson bench prototype\nP4 enclosure mock-up",
              "P1 laptop demonstrator"]
    ys = [H - 0.35 - bh - i * 1.62 for i in range(4)]
    xl = [0.15 + i * 0.5 for i in range(4)]
    xr = [W - 0.15 - bw - i * 0.5 for i in range(4)]
    for i in range(4):
        box(ax, xl[i], ys[i], bw, bh, left[i][0], [left[i][1]], title_size=ts, line_size=ls)
        box(ax, xr[i], ys[i], bw, bh, right[i][0], [right[i][1]], title_size=ts, line_size=ls, fc=G4)
        # horizontal link: the right side checks the left side
        line(ax, [(xl[i] + bw, ys[i] + bh / 2), (xr[i], ys[i] + bh / 2)], lw=0.9, ls=(0, (4, 2.5)), z=1)
        ax.text((xl[i] + bw + xr[i]) / 2, ys[i] + bh / 2 + 0.06, "validates" if i == 0 else "verifies",
                ha="center", va="bottom", fontsize=7.3, style="italic", color="#333333")
        ax.text((xl[i] + bw + xr[i]) / 2, ys[i] + bh / 2 - 0.08, protos[i], ha="center", va="top", fontsize=7.2,
                color="#222222", linespacing=1.15)
        if i < 3:
            arrow(ax, xl[i] + bw * 0.35, ys[i], xl[i + 1] + bw * 0.35, ys[i + 1] + bh)
            arrow(ax, xr[i + 1] + bw * 0.65, ys[i + 1] + bh, xr[i] + bw * 0.65, ys[i])
    # bottom of the V
    ib_w = 4.4
    yb = ys[3] - 1.55
    box(ax, W / 2 - ib_w / 2, yb, ib_w, 1.05, "Implementation", ["code, trained models, wiring"], title_size=ts,
        line_size=ls, fc=G3)
    arrow(ax, xl[3] + bw * 0.35, ys[3], W / 2 - ib_w / 2 + 0.3, yb + 1.05)
    arrow(ax, W / 2 + ib_w / 2 - 0.3, yb + 1.05, xr[3] + bw * 0.65, ys[3])
    ax.text(0.2, yb + 0.5, "design and\ndecomposition", ha="left", va="center", fontsize=7.8, style="italic")
    ax.text(W - 0.2, yb + 0.5, "testing and\nintegration", ha="right", va="center", fontsize=7.8, style="italic")
    fig.savefig(os.path.join(HERE, "fig_vmodel.png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    want = sys.argv[1:] or ["flow", "block", "results", "sim", "vmodel"]
    for name in want:
        globals()["fig_" + name]()
        print("drew", name)
