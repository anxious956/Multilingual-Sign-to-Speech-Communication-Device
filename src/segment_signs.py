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
import warnings
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


def shoulder_width(frames):
    """Shoulder width per frame, carried over frames where the body is lost."""
    w = np.linalg.norm(frames[:, L_SHOULDER, :2] - frames[:, R_SHOULDER, :2], axis=1)
    ok = ~np.isnan(w)
    if not ok.any():
        return np.ones(len(w))
    idx = np.where(ok, np.arange(len(w)), 0)
    np.maximum.accumulate(idx, out=idx)
    w = w[idx]
    w[: ok.argmax()] = w[ok.argmax()]
    return w


def hand_speed(frames, smooth=3):
    """Per frame: how fast the faster hand moves, in shoulder widths per frame (0 if no hand)."""
    speed = np.zeros(len(frames))
    for start in (LEFT, RIGHT):
        hand = frames[:, start:start + 21, :2]
        step = np.linalg.norm(np.diff(hand, axis=0), axis=2)  # (T-1, 21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames without this hand
            s = np.nanmean(step, axis=1)
        speed[1:] = np.fmax(speed[1:], np.nan_to_num(s))
    speed /= shoulder_width(frames)
    return np.convolve(speed, np.ones(smooth) / smooth, mode="same")


def split_at_pauses(segs, speed, min_frames=8, pause=0.35):
    """Split each segment where the hands almost stop between two signs (fluent signing, no rest).

    A cut is placed at a local minimum of the hand speed that is below `pause` times the
    segment's median speed and at least min_frames from the segment ends and the previous cut.
    """
    out = []
    for s, e in segs:
        v = speed[s:e]
        limit = pause * np.median(v)
        cuts, last = [], 0
        for i in range(min_frames, len(v) - min_frames):
            if v[i] < limit and v[i] <= v[i - 1] and v[i] <= v[i + 1] and i - last >= min_frames:
                cuts.append(i)
                last = i
        bounds = [0] + cuts + [len(v)]
        out += [(s + a, s + b) for a, b in zip(bounds[:-1], bounds[1:])]
    return out


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
    parser.add_argument("--pauses", action="store_true",
                        help="also split where the hands almost stop, for signs made without lowering the hands")
    parser.add_argument("--pause", type=float, default=0.35,
                        help="with --pauses: a pause is slower than this times the median hand speed (default: 0.35)")
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
        timeline = ["#" if u else "." for u in up]
        if args.pauses:
            segs = split_at_pauses(segs, hand_speed(rec), args.min_frames, args.pause)
            for s, _ in segs[1:]:
                if timeline[s] == "#":
                    timeline[s] = "|"  # a cut between two signs inside one raised stretch
        timeline = "".join(timeline)
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
