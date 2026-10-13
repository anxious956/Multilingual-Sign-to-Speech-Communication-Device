"""Play back recordings saved by capture_landmarks.py as a moving skeleton.

Usage:
    python src/view_recording.py data/recordings/HELLO                # every recording in the folder
    python src/view_recording.py data/recordings/HELLO/HELLO_xxx.npy  # one recording

Keys:
    SPACE  pause / play
    N      next recording
    Q      quit
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

W, H = 640, 480
FACE, LEFT, POSE, RIGHT = 0, 468, 489, 522  # first row of each part (Kaggle ISLR order)
HOL = mp.solutions.holistic
PARTS = [  # (first row, connections, colour BGR)
    (FACE, HOL.FACEMESH_CONTOURS, (200, 200, 200)),
    (POSE, HOL.POSE_CONNECTIONS, (255, 255, 255)),
    (LEFT, HOL.HAND_CONNECTIONS, (0, 200, 255)),   # signer's left hand: orange
    (RIGHT, HOL.HAND_CONNECTIONS, (255, 150, 0)),  # signer's right hand: blue
]


def to_px(p):
    # mirror x so the playback looks like the capture preview
    return int((1 - p[0]) * W), int(p[1] * H)


def draw(points):
    img = np.zeros((H, W, 3), np.uint8)
    for start, connections, colour in PARTS:
        for a, b in connections:
            pa, pb = points[start + a], points[start + b]
            if not (np.isnan(pa[:2]).any() or np.isnan(pb[:2]).any()):
                cv2.line(img, to_px(pa), to_px(pb), colour, 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a .npy recording or a folder of them")
    parser.add_argument("--fps", type=float, default=20, help="playback speed (default: 20)")
    args = parser.parse_args()

    path = Path(args.path)
    files = sorted(path.glob("*.npy")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No .npy files in {path}")

    delay = max(int(1000 / args.fps), 1)
    for f in files:
        rec = np.load(f)
        hands = (~np.isnan(rec[:, LEFT:POSE, 0]).all(1)) | (~np.isnan(rec[:, RIGHT:, 0]).all(1))
        print(f"{f.name}: {len(rec)} frames, a hand detected in {hands.sum()} of them")
        i, paused = 0, False
        while True:
            img = draw(rec[i])
            cv2.putText(img, f"{f.name}  frame {i + 1}/{len(rec)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Recording (SPACE: pause, N: next, Q: quit)", img)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return
            if key in (ord("n"), ord("N")):
                break
            if key == ord(" "):
                paused = not paused
            if not paused:
                i = (i + 1) % len(rec)  # loop until N is pressed
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
