"""Split several signs made without lowering the hands, with help from the sign model.

The hand-speed rule in segment_signs.py finds where the hands slow down, but
the hands also stop inside a sign (HELLO pauses at the forehead, WATER at the
chin), so speed alone cuts signs in half and misses real boundaries. Here the
slow-downs are only candidate cuts; every way of cutting at them is scored by
the sign model, and the cuts whose pieces the model recognises most confidently
are kept (the same idea as word segmentation in speech recognition).

Score of a cut = sum over its pieces of (log probability of the best sign + word bonus).
A larger --word-bonus favours more, shorter pieces.

Usage:
    python src/fluent_segment.py data/recordings/FLUENT --model models/islr_model_v2.pt --vocab hello,water,please,...
"""
import argparse
from pathlib import Path

import numpy as np

from segment_signs import hand_speed, raised, segments
from sign_model import SignClassifier


def candidate_cuts(speed, pause=0.6, spacing=4):
    """Frames where the hands slow down: local speed minima below pause x the median speed."""
    limit = pause * np.median(speed)
    cuts = []
    for i in range(1, len(speed) - 1):
        if speed[i] < limit and speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]:
            if not cuts or i - cuts[-1] >= spacing:
                cuts.append(i)
    return cuts


def best_split(frames, classifier, cuts, min_len=8, max_len=60, word_bonus=0.5):
    """Choose the cuts whose pieces the model is most sure of. Returns [(start, end, sign, prob), ...]."""
    points = [0] + [c for c in cuts if min_len <= c <= len(frames) - min_len] + [len(frames)]
    cache = {}

    def piece(i, j):
        if (i, j) not in cache:
            sign, p = classifier.predict(frames[points[i]:points[j]], top=1)[0]
            cache[i, j] = (np.log(max(p, 1e-9)) + word_bonus, sign, p)
        return cache[i, j]

    best = [-np.inf] * len(points)
    back = [None] * len(points)
    best[0] = 0.0
    for j in range(1, len(points)):
        for i in range(j):
            length = points[j] - points[i]
            if best[i] == -np.inf or length < min_len or (length > max_len and j - i > 1):
                continue
            score = best[i] + piece(i, j)[0]
            if score > best[j]:
                best[j], back[j] = score, i
    if back[-1] is None:  # too short to split: one piece
        sign, p = classifier.predict(frames, top=1)[0]
        return [(0, len(frames), sign, p)]
    out, j = [], len(points) - 1
    while j:
        i = back[j]
        _, sign, p = piece(i, j)
        out.append((points[i], points[j], sign, p))
        j = i
    return out[::-1]


def merge_repeats(pieces):
    """Join neighbouring pieces given the same sign: one sign cut in two (e.g. at its own hold)."""
    out = []
    for a, b, sign, p in pieces:
        if out and out[-1][2] == sign:
            out[-1] = (out[-1][0], b, sign, max(out[-1][3], p))
        else:
            out.append((a, b, sign, p))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a .npy recording or a folder of them")
    parser.add_argument("--model", default="models/islr_model_v2.pt")
    parser.add_argument("--vocab", help="only these signs, comma separated")
    parser.add_argument("--x-scale", type=float, default=2.08, help="see live_demo.py (default: 2.08)")
    parser.add_argument("--pause", type=float, default=0.6,
                        help="candidate cuts where the speed is below this times the median (default: 0.6)")
    parser.add_argument("--word-bonus", type=float, default=0.5, help="larger = more pieces (default: 0.5)")
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.x_scale)
    if args.vocab:
        classifier.limit_to([w.strip() for w in args.vocab.split(",")])
    path = Path(args.path)
    files = sorted(path.glob("*.npy")) if path.is_dir() else [path]

    for f in files:
        rec = np.load(f)
        speed = hand_speed(rec)
        print(f"{f.name}: {len(rec)} frames")
        for s, e in segments(raised(rec)):  # stretches with the hands up, separated by rest
            stretch = rec[s:e]
            cuts = candidate_cuts(speed[s:e], args.pause)
            whole = classifier.predict(stretch, top=1)[0]
            pieces = merge_repeats(best_split(stretch, classifier, cuts, word_bonus=args.word_bonus))
            print(f"  frames {s}-{e}: {len(cuts)} candidate cuts; as one sign: {whole[0]} {whole[1]:.0%}")
            print(f"    candidate cuts at frames {[s + c for c in cuts]}")
            print("    -> " + "  |  ".join(f"{sign} {p:.0%} ({s + a}-{s + b})" for a, b, sign, p in pieces))


if __name__ == "__main__":
    main()
