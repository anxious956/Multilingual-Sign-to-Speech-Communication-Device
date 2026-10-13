"""Find where each sign starts and ends in a landmark recording (Goal 1, step 2).

A sign is taken to be happening while at least one hand is raised; the
signer's hands at rest (lowered or out of the frame) separate signs. A hand
counts as raised when it is detected and its wrist is above a "rest line"
drawn below the shoulders. The rest line is measured in shoulder widths so
it does not depend on how far the signer sits from the camera.

Short gaps and flickers are smoothed out: a sign must last at least
--min-frames frames, and the hands must be at rest for at least --rest-frames
frames before a sign is considered finished.

Usage:
    python src/segment_signs.py data/recordings/HELLO           # report the signs found in each recording
    python src/segment_signs.py data/recordings/HELLO --trim    # also save each sign to data/segments/<LABEL>/
"""
import argparse
from pathlib import Path

import numpy as np

LEFT, POSE, RIGHT = 468, 489, 522  # first row of each part (Kaggle ISLR order)
L_SHOULDER, R_SHOULDER = POSE + 11, POSE + 12
WRISTS = (LEFT, RIGHT)  # landmark 0 of each hand is the wrist
REPO = Path(__file__).resolve().parent.parent


def raised(frames, rest_line=1.0):
    """Per frame: True if at least one hand is detected and above the rest line.

    rest_line is how far below the shoulders the line sits, in shoulder widths.
    """
    shoulders = frames[:, [L_SHOULDER, R_SHOULDER], :2]  # (T, 2, xy)
    mid_y = shoulders[:, :, 1].mean(1)
    width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    # if the body is lost for a moment, keep the last known shoulders
    for arr in (mid_y, width):
        ok = ~np.isnan(arr)
        if ok.any():
            idx = np.where(ok, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            arr[:] = arr[idx]
            arr[: ok.argmax()] = arr[ok.argmax()]
    line = mid_y + rest_line * width
    line[np.isnan(line)] = np.inf  # no body found at all: any detected hand counts as raised

    up = np.zeros(len(frames), bool)
    for wrist in WRISTS:
        y = frames[:, wrist, 1]
        up |= ~np.isnan(y) & (y < line)  # image y grows downwards
    return up


def segments(up, min_frames=8, rest_frames=6, pad=3):
    """Turn per-frame raised flags into (start, end) frame ranges, end exclusive."""
    found, start, rest = [], None, 0
    for i, u in enumerate(up):
        if u:
            if start is None:
                start = i
            rest = 0
        elif start is not None:
            rest += 1
            if rest >= rest_frames:
                found.append((start, i - rest + 1))
                start, rest = None, 0
    if start is not None:
        found.append((start, len(up) - rest))
    return [(max(s - pad, 0), min(e + pad, len(up))) for s, e in found if e - s >= min_frames]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a .npy recording or a folder of them")
    parser.add_argument("--rest-line", type=float, default=1.0,
                        help="rest line below the shoulders, in shoulder widths (default: 1.0)")
    parser.add_argument("--min-frames", type=int, default=8, help="shortest sign, in frames (default: 8)")
    parser.add_argument("--rest-frames", type=int, default=6,
                        help="frames at rest that end a sign (default: 6)")
    parser.add_argument("--trim", action="store_true", help="save each sign found to data/segments/<LABEL>/")
    args = parser.parse_args()

    path = Path(args.path)
    files = sorted(path.glob("*.npy")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No .npy files in {path}")

    for f in files:
        rec = np.load(f)
        up = raised(rec, args.rest_line)
        segs = segments(up, args.min_frames, args.rest_frames)
        timeline = "".join("#" if u else "." for u in up)
        print(f"{f.name}: {len(rec)} frames, {len(segs)} sign(s) {segs}")
        print(f"  {timeline}")
        if args.trim:
            label = f.parent.name
            out = REPO / "data" / "segments" / label
            out.mkdir(parents=True, exist_ok=True)
            for k, (s, e) in enumerate(segs):
                np.save(out / f"{f.stem}_{k}.npy", rec[s:e])
    if args.trim:
        print(f"saved to {REPO / 'data' / 'segments'}")


if __name__ == "__main__":
    main()
