# Goal 1, step 3a: shrink the Kaggle Google ISLR data (56 GB) to a few GB.
#
# Run this in a Kaggle Notebook created from the competition page
# (https://www.kaggle.com/competitions/asl-signs -> Code -> New Notebook),
# where the data is already attached; nothing is downloaded.
# Paste the whole file into one cell, then "Save Version -> Save & Run All".
# The output files appear in the notebook's Output tab:
#
#   islr_landmarks.npy  all frames of all signs, (total frames, N_KEEP, 3) float16, NaN = not detected
#   islr_meta.npz       per sign: offset, length, label, participant_id, sequence_id
#   islr_info.json      landmark rows kept and the sign name of each label
#
# Sign i is landmarks[offset[i] : offset[i] + length[i]].
# Only lips, both hands and the upper body are kept; the rest of the face is
# dropped. The rows kept use the same 543-landmark order as our own
# recordings (src/capture_landmarks.py), so KEEP can be applied to them too.
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---- where the competition data is ----
candidates = [os.environ.get("DATA", ""), "/kaggle/input/asl-signs", "/kaggle/input/competitions/asl-signs"]
candidates += [os.path.dirname(p) for p in glob.glob("/kaggle/input/*/train.csv") + glob.glob("/kaggle/input/*/*/train.csv")]
ROOT = next(c for c in candidates if os.path.exists(os.path.join(c, "train.csv")))
OUT = os.environ.get("OUT", "/kaggle/working")
print("data:", ROOT)

# ---- landmarks to keep (rows of the 543-landmark frame) ----
FACE, LEFT, POSE, RIGHT = 0, 468, 489, 522
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
UPPER_BODY = range(11, 23)  # pose: shoulders, elbows, wrists and the hand points of the pose model
KEEP = ([FACE + i for i in LIPS] + list(range(LEFT, LEFT + 21))
        + [POSE + i for i in UPPER_BODY] + list(range(RIGHT, RIGHT + 21)))
print("landmarks kept:", len(KEEP))  # 40 + 21 + 12 + 21 = 94

train = pd.read_csv(os.path.join(ROOT, "train.csv"))
with open(os.path.join(ROOT, "sign_to_prediction_index_map.json")) as f:
    sign_to_label = json.load(f)
labels = train["sign"].map(sign_to_label).to_numpy(np.int16)
print(len(train), "signs,", len(sign_to_label), "classes,", train["participant_id"].nunique(), "signers")

# every file stores each frame as 543 rows in the order face, left_hand, pose, right_hand
first = pd.read_parquet(os.path.join(ROOT, train["path"][0]))
types = first["type"].iloc[:543]
assert (types.iloc[:468] == "face").all() and (types.iloc[468:489] == "left_hand").all()
assert (types.iloc[489:522] == "pose").all() and (types.iloc[522:] == "right_hand").all()


def load(path):
    xyz = pd.read_parquet(os.path.join(ROOT, path), columns=["x", "y", "z"]).to_numpy(np.float32)
    return xyz.reshape(-1, 543, 3)[:, KEEP].astype(np.float16)


start = time.time()
signs = []
CHUNK = 5000
for i in range(0, len(train), CHUNK):
    signs += Parallel(n_jobs=os.cpu_count())(delayed(load)(p) for p in train["path"][i:i + CHUNK])
    print(f"{len(signs)}/{len(train)} signs read, {time.time() - start:.0f} s")

lengths = np.array([len(s) for s in signs], np.int32)
offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
landmarks = np.concatenate(signs)
del signs
np.save(os.path.join(OUT, "islr_landmarks.npy"), landmarks)
np.savez(os.path.join(OUT, "islr_meta.npz"), offset=offsets, length=lengths, label=labels,
         participant_id=train["participant_id"].to_numpy(np.int32),
         sequence_id=train["sequence_id"].to_numpy(np.int64))
with open(os.path.join(OUT, "islr_info.json"), "w") as f:
    json.dump({"keep": KEEP, "signs": sorted(sign_to_label, key=sign_to_label.get)}, f)

# ---- quick summary ----
hand_seen = ~np.isnan(landmarks[:, 40:61, 0]).all(1) | ~np.isnan(landmarks[:, 73:94, 0]).all(1)
print(f"saved {landmarks.shape} ({landmarks.nbytes / 1e9:.2f} GB) to {OUT}")
print(f"frames per sign: median {np.median(lengths):.0f}, 95% under {np.percentile(lengths, 95):.0f}, max {lengths.max()}")
print(f"frames with at least one hand: {hand_seen.mean():.0%}")
print("hello" in sign_to_label and f"'hello' is label {sign_to_label['hello']}" or "'hello' is not in the 250 signs")
