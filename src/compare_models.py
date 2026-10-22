"""Run several trained models on saved recordings, also slowed down in time, to compare them offline.

Each recording (a .npy from capture_landmarks.py or live_demo.py --save) is classified as one sign by every
model, as recorded and resampled to k times as many frames (k = 1.5, 2, 3 ...). Resampling repeats frames the
same way as the speed augmentation in training, so k = 2 makes a clip recorded at 15 fps look like 30 fps. If a
model only works on the slowed-down clips, the live pipeline runs at a lower frame rate than its training data.

Usage (Windows cmd; quote nothing, patterns with * are expanded here):
    python src\\compare_models.py data\\recordings\\LIVE\\LIVE_20261021_2342*.npy ^
        --models v4=models\\islr_model_v4.pt:0.84 v5=models\\asl_citizen_model_v5.pt:0.55 ^
        --labels hello,hello,water,water,thankyou,thankyou,please,please
--labels gives the true sign of each recording in file-name order (= time order) and adds an accuracy table.
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from sign_model import HANDS, SignClassifier


def recordings(patterns):
    files = []
    for p in patterns:
        if Path(p).is_dir():
            files += sorted(Path(p).rglob("*.npy"))
        else:
            files += sorted(Path(f) for f in glob.glob(p))
    return files


def slow_down(rec, k):
    """Resample to round(k * frames) frames by repeating frames (floor index), like the training augmentation."""
    n = len(rec)
    m = max(2, round(n * k))
    return rec[np.minimum((np.arange(m) * n / m).astype(int), n - 1)]


def frames_with_hand(rec):
    """Frames where MediaPipe found at least one hand (rec in the 543-landmark order)."""
    left = ~np.isnan(rec[:, 468:489, 0]).all(1)
    right = ~np.isnan(rec[:, 522:543, 0]).all(1)
    return int((left | right).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help=".npy files, patterns with * or folders")
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path[:temperature], e.g. v5=models\\asl_citizen_model_v5.pt:0.55")
    parser.add_argument("--speeds", default="1,1.5,2,3", help="slow-down factors (default: 1,1.5,2,3)")
    parser.add_argument("--labels", help="true sign of each recording, comma separated, in file-name order")
    parser.add_argument("--vocab", help="only these signs, comma separated (applied to every model)")
    parser.add_argument("--x-scale", type=float, default=2.08, help="for v1-v3 models only (see live_demo.py)")
    args = parser.parse_args()

    files = recordings(args.paths)
    if not files:
        raise SystemExit("no recordings found")
    labels = [w.strip() for w in args.labels.split(",")] if args.labels else None
    if labels and len(labels) != len(files):
        raise SystemExit(f"{len(labels)} labels for {len(files)} recordings:\n" + "\n".join(f.name for f in files))
    speeds = [float(s) for s in args.speeds.split(",")]
    models = {}
    for item in args.models:
        name, spec = item.split("=", 1)
        path, _, t = spec.rpartition(":") if spec.rpartition(":")[2].replace(".", "", 1).isdigit() else (spec, "", "")
        clf = SignClassifier(path, args.x_scale, float(t) if t else 1.0)
        if args.vocab:
            clf.limit_to([w.strip() for w in args.vocab.split(",")])
        models[name] = clf

    right = {(name, k): 0 for name in models for k in speeds}
    for i, f in enumerate(files):
        rec = np.load(f).astype(np.float32)
        head = f"{f.name}  {len(rec)} frames, {frames_with_hand(rec)} with a hand" + (f"  [{labels[i]}]" if labels else "")
        print(head)
        for name, clf in models.items():
            cells = []
            for k in speeds:
                sign, p = clf.predict(slow_down(rec, k), top=1)[0]
                ok = labels is not None and sign == labels[i]
                right[(name, k)] += ok
                cells.append(f"x{k:g} {sign} {p:.0%}" + (" ✓" if ok else ""))
            print(f"   {name:>4}: " + "  |  ".join(cells))
    if labels:
        print(f"\nCorrect top-1 out of {len(files)}:")
        print("| Model | " + " | ".join(f"x{k:g}" for k in speeds) + " |")
        print("|---|" + "---|" * len(speeds))
        for name in models:
            print(f"| {name} | " + " | ".join(str(right[(name, k)]) for k in speeds) + " |")


if __name__ == "__main__":
    main()
