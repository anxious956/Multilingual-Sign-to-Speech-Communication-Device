"""Extract MediaPipe Holistic landmarks from ASL Citizen videos, in the same format as our Kaggle data.

Videos are read one by one straight from Microsoft's zip (HTTP range requests through `remotezip`),
so the 42.8 GB archive is never downloaded as a whole. Each video goes through MediaPipe Holistic
(same version and settings as src/capture_landmarks.py) and the same 94 landmarks as
notebooks/01_preprocess_islr.py are kept (lips, both hands, upper body), as float16.

Work is split into chunks of videos; finished chunks are skipped, so a run can be stopped and resumed,
and several machines can share the work with --shard i --shards N. --merge joins the chunks into
  asl_citizen_landmarks.npy  (total frames, 94, 3) float16, NaN = not detected
  asl_citizen_meta.npz       offset, length, label, participant_id, split (0 train, 1 val, 2 test), fps, video
  asl_citizen_info.json      keep, signs (gloss names, label order), signers
ASL Citizen may not be redistributed (Microsoft Research License): keep these files in the team Drive.

Needs Python 3.9-3.12 (mediapipe 0.10.14). On Colab (Python 3.13) make a 3.11 environment first:
    !pip install -q uv
    !uv venv /content/mp --python 3.11 -q
    !uv pip install -q --python /content/mp mediapipe==0.10.14 "numpy<2" opencv-python-headless remotezip
    !/content/mp/bin/python tools/extract_asl_citizen.py --out /content/drive/MyDrive/sign2speech/asl_citizen --glosses all
    !/content/mp/bin/python tools/extract_asl_citizen.py --out /content/drive/MyDrive/sign2speech/asl_citizen --merge
"""
import argparse
import csv
import io
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Colab sets MPLBACKEND to its notebook backend, which the separate venv's matplotlib (imported by mediapipe) rejects
os.environ["MPLBACKEND"] = "Agg"

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
SPLITS = {"train": 0, "val": 1, "test": 2}
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
KEEP = LIPS + list(range(468, 489)) + [489 + i for i in range(11, 23)] + list(range(522, 543))  # = 01_preprocess_islr.py
MAX_SIDE = 640  # frames are shrunk to this size before MediaPipe (speed; the landmarks are relative anyway)

_holistic = None


def _init_worker():
    global _holistic
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["GLOG_minloglevel"] = "2"
    import mediapipe as mp
    _holistic = mp.solutions.holistic.Holistic(model_complexity=1)


def _to_array(lms, n):
    if lms is None:
        return np.full((n, 3), np.nan, np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)


def landmarks_of(path):
    """(frames, 94, 3) float16 for one video file, and its frame rate."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        r = _holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        full = np.concatenate([_to_array(r.face_landmarks, 468), _to_array(r.left_hand_landmarks, 21),
                               _to_array(r.pose_landmarks, 33), _to_array(r.right_hand_landmarks, 21)])
        frames.append(full[KEEP])
    cap.release()
    arr = np.stack(frames).astype(np.float16) if frames else np.zeros((0, len(KEEP), 3), np.float16)
    return arr, fps


def _work(item):
    path = item
    try:
        arr, fps = landmarks_of(path)
    finally:
        os.remove(path)
    return arr, fps


_local = threading.local()


def _fetch(member):
    """Read one zip member over HTTP; each download thread keeps its own connection."""
    from remotezip import RemoteZip
    if not hasattr(_local, "zip"):
        _local.zip = RemoteZip(URL)
    return _local.zip.read(member)


def read_rows(z):
    rows = []
    for split in SPLITS:
        f = io.TextIOWrapper(z.open(f"ASL_Citizen/splits/{split}.csv"), encoding="utf-8")
        for r in csv.DictReader(f):
            rows.append((r["Participant ID"], r["Video file"], r["Gloss"], split))
    return rows


def extract(args):
    from remotezip import RemoteZip
    out = Path(args.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        rows = read_rows(z)
    if args.glosses != "all":
        wanted = set(g.strip().upper() for g in Path(args.glosses).read_text().replace(",", "\n").split()) \
            if os.path.exists(args.glosses) else set(g.strip().upper() for g in args.glosses.split(","))
        rows = [r for r in rows if r[2] in wanted]
    rows = [r for i, r in enumerate(rows) if i % args.shards == args.shard]
    chunks = [rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk)]
    print(f"{len(rows)} videos in this shard, {len(chunks)} chunks of {args.chunk}, {args.workers} workers")
    tmp = Path(tempfile.mkdtemp())
    todo = [(k, c) for k, c in enumerate(chunks)
            if not (out / "chunks" / f"{args.tag}_s{args.shard}of{args.shards}_c{k:04d}.npz").exists()]
    print(f"{len(chunks) - len(todo)} chunks already done")

    def download(k, chunk):  # write the chunk's videos to temporary files, several at a time
        datas = list(fetcher.map(_fetch, [f"ASL_Citizen/videos/{r[1]}" for r in chunk]))
        paths = []
        for j, data in enumerate(datas):
            p = tmp / f"{k}_{j}.mp4"
            p.write_bytes(data)
            paths.append(str(p))
        return paths

    with Pool(args.workers, initializer=_init_worker) as pool, ThreadPoolExecutor(args.downloads) as fetcher, \
            ThreadPoolExecutor(1) as prefetch:
        nxt = prefetch.submit(download, *todo[0]) if todo else None
        for n, (k, chunk) in enumerate(todo):
            t0 = time.time()
            paths = nxt.result()
            nxt = prefetch.submit(download, *todo[n + 1]) if n + 1 < len(todo) else None  # overlap the next download
            t1 = time.time()
            results = pool.map(_work, paths, chunksize=1)
            arrs = [a for a, _ in results]
            lengths = np.array([len(a) for a in arrs], np.int32)
            np.savez(out / "chunks" / f"{args.tag}_s{args.shard}of{args.shards}_c{k:04d}.npz",
                     landmarks=np.concatenate(arrs), length=lengths,
                     fps=np.array([f for _, f in results], np.float32),
                     participant=np.array([r[0] for r in chunk]), video=np.array([r[1] for r in chunk]),
                     gloss=np.array([r[2] for r in chunk]), split=np.array([SPLITS[r[3]] for r in chunk], np.int8))
            print(f"chunk {k + 1}/{len(chunks)}: {len(chunk)} videos, {lengths.sum()} frames, "
                  f"waited {t1 - t0:.0f} s for download, landmarks {time.time() - t1:.0f} s", flush=True)


def merge(args):
    out = Path(args.out)
    files = sorted((out / "chunks").glob("*.npz"))
    parts = [dict(np.load(f)) for f in files]
    meta = {k: np.concatenate([p[k] for p in parts]) for k in ("length", "fps", "participant", "video", "gloss", "split")}
    # one entry per video even if several runs overlapped
    _, first = np.unique(meta["video"], return_index=True)
    first = np.sort(first)
    landmarks, offsets, cursor = [], [], 0
    starts = np.concatenate([[0], np.cumsum(np.concatenate([p["length"] for p in parts]))[:-1]])
    all_lm = np.concatenate([p["landmarks"] for p in parts])
    for i in first:
        landmarks.append(all_lm[starts[i]:starts[i] + meta["length"][i]])
        offsets.append(cursor)
        cursor += meta["length"][i]
    signs = sorted(set(meta["gloss"][first]))
    signers = sorted(set(meta["participant"][first]), key=lambda s: int(s.lstrip("P")) if s.lstrip("P").isdigit() else s)
    np.save(out / "asl_citizen_landmarks.npy", np.concatenate(landmarks))
    np.savez(out / "asl_citizen_meta.npz", offset=np.array(offsets, np.int64), length=meta["length"][first],
             label=np.array([signs.index(g) for g in meta["gloss"][first]], np.int16),
             participant_id=np.array([signers.index(p) for p in meta["participant"][first]], np.int32),
             split=meta["split"][first], fps=meta["fps"][first], video=meta["video"][first])
    with open(out / "asl_citizen_info.json", "w") as f:
        json.dump({"keep": KEEP, "signs": signs, "signers": signers}, f)
    hand = np.concatenate(landmarks)
    seen = ~np.isnan(hand[:, 40:61, 0]).all(1) | ~np.isnan(hand[:, 73:94, 0]).all(1)
    print(f"{len(first)} videos, {len(signs)} signs, {len(signers)} signers, {cursor} frames; "
          f"frames with a hand: {seen.mean():.0%}; median fps {np.median(meta['fps'][first]):.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output folder (e.g. on Google Drive)")
    parser.add_argument("--glosses", default="all", help="'all', comma separated glosses, or a file listing them")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1, help="split the work over several machines")
    parser.add_argument("--chunk", type=int, default=400, help="videos per saved chunk")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel MediaPipe processes")
    parser.add_argument("--downloads", type=int, default=8, help="parallel video downloads")
    parser.add_argument("--tag", default="run", help="name prefix for the chunk files (use different tags for different gloss lists)")
    parser.add_argument("--merge", action="store_true", help="join the finished chunks into the final files")
    args = parser.parse_args()
    merge(args) if args.merge else extract(args)


if __name__ == "__main__":
    main()
