"""Calibration and abstention analysis for a trained sign model (Goal 1, step 5).

Uses the validation and test outputs saved by the training notebooks
(islr_val_logits.npz from 02, islr_v3_logits.npz from 03); no training needed.
Everything is fitted on the validation signers and reported on the test signers.

What it reports:
  - accuracy: top-1 / top-5, per test signer, signer average, worst signer
  - calibration: temperature T, NLL, ECE before and after temperature scaling, reliability diagram
  - abstention: risk-coverage curve, AURC, and the confidence threshold that keeps the error of
    accepted signs at 5% / 10% on validation, with the coverage and error it gives on test
  - candidate lists: a split-conformal set that contains the right sign 90% of the time, its size
  - with --vocab: the same for a smaller vocabulary, plus how well the confidence separates
    in-vocabulary signs from signs outside it (AUROC, FPR at 95% TPR)

Usage:
    python src/calibrate.py --logits islr_val_logits.npz --meta islr_meta.npz --info islr_info.json
    python src/calibrate.py ... --vocab hello,bye,please,thankyou,yes,no,water --out docs/figures/calib_v2
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEED, VAL_SIGNERS, TEST_SIGNERS = 42, 3, 3  # must match the training notebooks


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z, y, T=1.0):
    p = softmax(z, T)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()


def fit_temperature(z, y):
    """T minimising validation NLL (search on log T; NLL is smooth and has one minimum in T)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    T = grid[np.argmin([nll(z, y, t) for t in grid])]
    for step in (0.02, 0.005, 0.001):  # refine around the best grid point
        fine = T * np.exp(np.linspace(-20 * step, 20 * step, 41))
        T = fine[np.argmin([nll(z, y, t) for t in fine])]
    return float(T)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def risk_coverage(conf, correct):
    """Sort by confidence; coverage k/n and error among the k most confident."""
    order = np.argsort(-conf)
    errors = np.cumsum(~correct[order])
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), errors / k, conf[order]


def threshold_for_risk(conf, correct, target):
    """Lowest confidence threshold whose accepted signs have error <= target (on this data)."""
    cov, risk, sorted_conf = risk_coverage(conf, correct)
    ok = np.where(risk <= target)[0]
    return float(sorted_conf[ok.max()]) if len(ok) else 1.0


def auroc(pos, neg):
    """Probability that a random positive scores above a random negative."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def signer_ids(meta_path):
    meta = np.load(meta_path)
    signers = np.unique(meta["participant_id"])
    held = np.random.default_rng(SEED).choice(signers, VAL_SIGNERS + TEST_SIGNERS, replace=False)
    pid = meta["participant_id"]
    return pid[np.isin(pid, held[:VAL_SIGNERS])], pid[np.isin(pid, held[VAL_SIGNERS:])]


def analyse(name, zv, yv, zt, yt, st, out, signs):
    lines = [f"## {name}", ""]
    T = fit_temperature(zv, yv)
    pt_raw, pt = softmax(zt), softmax(zt, T)
    pred = pt.argmax(1)
    correct = pred == yt
    top5 = (np.argsort(-pt, 1)[:, :5] == yt[:, None]).any(1)
    conf_raw, conf = pt_raw.max(1), pt.max(1)

    per = {s: (correct[st == s].mean(), top5[st == s].mean(), (st == s).sum()) for s in np.unique(st)}
    lines += [f"Test signs: {len(yt)}, classes: {zt.shape[1]}", "",
              "| | Top-1 | Top-5 |", "|---|---|---|",
              f"| All test signs | {correct.mean():.1%} | {top5.mean():.1%} |",
              f"| Signer average | {np.mean([v[0] for v in per.values()]):.1%} | {np.mean([v[1] for v in per.values()]):.1%} |",
              f"| Worst signer | {min(v[0] for v in per.values()):.1%} | {min(v[1] for v in per.values()):.1%} |", ""]
    lines += ["| Test signer | Signs | Top-1 | Top-5 |", "|---|---|---|---|"]
    lines += [f"| {s} | {n} | {a:.1%} | {b:.1%} |" for s, (a, b, n) in per.items()]

    lines += ["", f"**Calibration** (temperature fitted on validation: T = {T:.2f}; T > 1 means the raw model was overconfident)", "",
              "| | NLL | ECE |", "|---|---|---|",
              f"| Raw softmax | {nll(zt, yt):.3f} | {ece(conf_raw, correct):.1%} |",
              f"| After temperature scaling | {nll(zt, yt, T):.3f} | {ece(conf, correct):.1%} |"]

    cov, risk, _ = risk_coverage(conf, correct)
    aurc = risk.mean()
    lines += ["", f"**Abstention** (threshold chosen on validation, measured on test). AURC = {aurc:.3f} (lower is better).", "",
              "| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |",
              "|---|---|---|---|"]
    conf_v, correct_v = softmax(zv, T).max(1), softmax(zv, T).argmax(1) == yv
    for target in (0.05, 0.10):
        tau = threshold_for_risk(conf_v, correct_v, target)
        acc = conf >= tau
        err = (~correct[acc]).mean() if acc.any() else float("nan")
        lines.append(f"| {target:.0%} | {tau:.2f} | {acc.mean():.1%} | {err:.1%} |")
        for s in np.unique(st):
            m = st == s
            a = acc & m
            lines.append(f"| ↳ signer {s} | | {a.sum() / m.sum():.1%} | {(~correct[a]).mean() if a.any() else float('nan'):.1%} |")

    # split conformal (LAC): set = signs with p >= 1 - qhat; covers the right sign >= 90% (on average)
    pv = softmax(zv, T)
    scores = 1 - pv[np.arange(len(yv)), yv]
    n = len(scores)
    qhat = np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher")
    sets = pt >= 1 - qhat
    covered = sets[np.arange(len(yt)), yt]
    size = sets.sum(1)
    lines += ["", f"**Candidate list** (split conformal, 90% target): test coverage {covered.mean():.1%}, "
              f"list size mean {size.mean():.1f}, median {np.median(size):.0f}; fixed top-5 covers {top5.mean():.1%}."]

    if out:
        plot(out, name, conf_raw, conf, correct, cov, risk)
        lines += ["", f"Figures: `{out}_{slug(name)}_reliability.png`, `{out}_{slug(name)}_risk_coverage.png`"]
    return lines, T


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def plot(out, name, conf_raw, conf, correct, cov, risk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for c, label, style in ((conf_raw, "raw", "o--"), (conf, "temperature scaled", "s-")):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (c > lo) & (c <= hi)
            if m.sum() >= 20:
                xs.append(c[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, style, label=label)
    ax.plot([0, 1], [0, 1], color="grey", lw=1)
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=f"Reliability: {name}", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}_{slug(name)}_reliability.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(cov, risk)
    ax.set(xlabel="Coverage (share of signs answered)", ylabel="Error among answered", title=f"Risk–coverage: {name}",
           xlim=(0, 1), ylim=(0, max(0.05, risk.max() * 1.05)))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_{slug(name)}_risk_coverage.png", dpi=150)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logits", required=True, help="islr_val_logits.npz (v1/v2) or islr_v3_logits.npz (v3)")
    parser.add_argument("--meta", required=True, help="islr_meta.npz, to know which signer made each sign")
    parser.add_argument("--info", required=True, help="islr_info.json, for the sign names")
    parser.add_argument("--vocab", help="comma separated signs: also analyse this smaller vocabulary")
    parser.add_argument("--out", help="prefix for the figures and the report, e.g. docs/figures/calib_v2")
    args = parser.parse_args()

    d = np.load(args.logits)
    signs = json.load(open(args.info))["signs"]
    zv, yv, zt, yt = d["val_logits"], d["val_labels"], d["test_logits"], d["test_labels"]
    sv, st = signer_ids(args.meta)
    assert len(sv) == len(yv) and len(st) == len(yt), "signer split does not match the saved outputs"

    lines, _ = analyse(f"All {zt.shape[1]} signs", zv, yv, zt, yt, st, args.out, signs)

    if args.vocab:
        words = [w.strip() for w in args.vocab.split(",")]
        missing = [w for w in words if w not in signs]
        if missing:
            raise SystemExit(f"not among the model's signs: {missing}")
        cols = np.array([signs.index(w) for w in words])
        remap = {c: i for i, c in enumerate(cols)}
        inv, int_ = np.isin(yv, cols), np.isin(yt, cols)
        zv_v, zt_v = zv[:, cols], zt[:, cols]
        yv_v = np.array([remap[c] for c in yv[inv]])
        yt_v = np.array([remap[c] for c in yt[int_]])
        vl, Tv = analyse(f"{len(words)}-sign vocabulary", zv_v[inv], yv_v, zt_v[int_], yt_v, st[int_], args.out, words)
        lines += [""] + vl
        # signs outside the vocabulary: the restricted softmax still picks one of the vocabulary signs
        p_in = softmax(zt_v[int_], Tv).max(1)
        p_out = softmax(zt_v[~int_], Tv).max(1)
        correct_in = softmax(zt_v[int_], Tv).argmax(1) == yt_v
        pos = p_in[correct_in]
        thr = np.quantile(pos, 0.05)
        lines += ["", f"**Signs outside the vocabulary** ({(~int_).sum()} test signs of the other {zt.shape[1] - len(words)} classes): "
                  f"AUROC (correct in-vocabulary vs outside) = {auroc(pos, p_out):.3f}; "
                  f"at the threshold keeping 95% of correct in-vocabulary signs, {(p_out >= thr).mean():.1%} of outside signs would still be accepted."]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(f"{args.out}_report.md").write_text(report + "\n", encoding="utf-8")
        print(f"\nsaved {args.out}_report.md")


if __name__ == "__main__":
    main()
