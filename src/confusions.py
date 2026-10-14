"""Which signs does the model mix up? Per-sign accuracy and the most frequent confusions on the test signers.

Uses the outputs saved by the training notebooks, like calibrate.py. Several models can be given to compare
them side by side. With --vocab, only those signs are considered (as in the live demo with --vocab).

Usage:
    python src/confusions.py --logits v2=models/islr_val_logits.npz v3=models/islr_v3_logits.npz --info models/islr_info.json --vocab hello,water,thankyou,food,...
    python src/confusions.py ... --pair thankyou,food
"""
import argparse
import json
from collections import Counter

import numpy as np


def predictions(path, signs, vocab):
    d = np.load(path)
    z, y = d["test_logits"], d["test_labels"]
    if vocab:
        cols = np.array([signs.index(w) for w in vocab])
        keep = np.isin(y, cols)
        z, y = z[keep][:, cols], y[keep]
        y = np.array([list(cols).index(c) for c in y])
        names = vocab
    else:
        names = signs
    return [names[i] for i in y], [names[i] for i in z.argmax(1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logits", nargs="+", required=True, help="name=path pairs, e.g. v2=models/islr_val_logits.npz")
    parser.add_argument("--info", required=True, help="islr_info.json")
    parser.add_argument("--vocab", help="comma separated signs to restrict to")
    parser.add_argument("--pair", help="two signs, e.g. thankyou,food: how often each is taken for the other")
    parser.add_argument("--top", type=int, default=15, help="how many confusions / weakest signs to list")
    args = parser.parse_args()

    signs = json.load(open(args.info))["signs"]
    vocab = [w.strip() for w in args.vocab.split(",")] if args.vocab else None
    if vocab and any(w not in signs for w in vocab):
        raise SystemExit(f"not among the model's signs: {[w for w in vocab if w not in signs]}")
    models = dict(item.split("=", 1) for item in args.logits)
    results = {name: predictions(path, signs, vocab) for name, path in models.items()}
    names = list(results)

    if args.pair:
        a, b = [w.strip() for w in args.pair.split(",")]
        print(f"| Model | {a} correct | {a} -> {b} | {b} correct | {b} -> {a} |")
        print("|---|---|---|---|---|")
        for name, (true, pred) in results.items():
            row = []
            for x, other in ((a, b), (b, a)):
                idx = [i for i, t in enumerate(true) if t == x]
                n = len(idx) or 1
                row += [f"{sum(pred[i] == x for i in idx) / n:.0%} of {len(idx)}", f"{sum(pred[i] == other for i in idx) / n:.0%}"]
            print(f"| {name} | " + " | ".join(row) + " |")
        print()

    # per-sign accuracy, weakest first (by the first model)
    per = {}
    for name, (true, pred) in results.items():
        total, right = Counter(true), Counter(t for t, p in zip(true, pred) if t == p)
        per[name] = {s: right[s] / total[s] for s in total}
    first = names[0]
    print(f"Weakest signs (test signers), sorted by {first}:")
    print("| Sign | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for s in sorted(per[first], key=per[first].get)[:args.top]:
        print(f"| {s} | " + " | ".join(f"{per[n].get(s, float('nan')):.0%}" for n in names) + " |")
    print()

    for name, (true, pred) in results.items():
        total = Counter(true)
        conf = Counter((t, p) for t, p in zip(true, pred) if t != p)
        print(f"Most frequent confusions, {name} (share of the true sign's test examples):")
        print("| True sign | Predicted as | Share |")
        print("|---|---|---|")
        for (t, p), c in sorted(conf.items(), key=lambda kv: -kv[1] / total[kv[0][0]])[:args.top]:
            print(f"| {t} | {p} | {c / total[t]:.0%} ({c} of {total[t]}) |")
        print()


if __name__ == "__main__":
    main()
