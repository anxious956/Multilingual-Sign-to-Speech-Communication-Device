"""Record ASL signs from a webcam as MediaPipe Holistic landmarks.

Each recording is saved as a NumPy array of shape (frames, 543, 3) holding
x, y, z for every landmark, in the same order as the Google Isolated Sign
Language Recognition (Kaggle) dataset, so our recordings and that dataset
can be used with the same model:

    rows   0-467  face        (468 points)
    rows 468-488  left hand   (21 points)
    rows 489-521  pose        (33 points)
    rows 522-542  right hand  (21 points)

Landmarks that MediaPipe does not detect in a frame are stored as NaN.

Usage:
    python src/capture_landmarks.py --label HELLO
    python src/capture_landmarks.py --video some_clip.mp4 --label HELLO   # process a file instead

Keys (webcam window):
    SPACE  start / stop recording one sign
    Q      quit
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

N_FACE, N_HAND, N_POSE = 468, 21, 33
N_LANDMARKS = N_FACE + N_HAND + N_POSE + N_HAND  # 543
# recordings go inside the repo folder (e.g. D:\Multilingual-...\data\recordings), wherever it is run from
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "recordings"


def to_array(landmarks, n):
    """Convert one MediaPipe landmark list to an (n, 3) array; NaN if not detected."""
    if landmarks is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)


def frame_landmarks(results):
    """All landmarks of one frame as a (543, 3) array in Kaggle ISLR order."""
    return np.concatenate([
        to_array(results.face_landmarks, N_FACE),
        to_array(results.left_hand_landmarks, N_HAND),
        to_array(results.pose_landmarks, N_POSE),
        to_array(results.right_hand_landmarks, N_HAND),
    ])


def save(frames, label, out_dir):
    """Save one recording to <out_dir>/<label>/<label>_<timestamp>.npy."""
    folder = Path(out_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_{datetime.now():%Y%m%d_%H%M%S}.npy"
    np.save(path, np.stack(frames))
    return path


def draw(image, results):
    drawing, styles, holistic = mp.solutions.drawing_utils, mp.solutions.drawing_styles, mp.solutions.holistic
    drawing.draw_landmarks(image, results.face_landmarks, holistic.FACEMESH_CONTOURS, None,
                           styles.get_default_face_mesh_contours_style())
    drawing.draw_landmarks(image, results.pose_landmarks, holistic.POSE_CONNECTIONS,
                           styles.get_default_pose_landmarks_style())
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        drawing.draw_landmarks(image, hand, holistic.HAND_CONNECTIONS,
                               styles.get_default_hand_landmarks_style(),
                               styles.get_default_hand_connections_style())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="UNLABELED", help="sign name used for the output folder, e.g. HELLO")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: data/recordings in the repo)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--video", help="process this video file instead of the webcam; the whole file is saved as one recording")
    parser.add_argument("--no-display", action="store_true", help="do not open a window (only with --video)")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + args.video if args.video else f'camera {args.camera}'}")

    recording = bool(args.video)  # a video file is recorded from start to end
    frames, saved = [], 0
    last = time.time()
    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # MediaPipe expects RGB; process the unmirrored frame so "left hand" is the signer's left hand
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if recording:
                frames.append(frame_landmarks(results))

            if args.no_display:
                continue
            draw(frame, results)
            view = cv2.flip(frame, 1)  # mirror only the preview, like a selfie camera
            now = time.time()
            fps, last = 1.0 / max(now - last, 1e-6), now
            hands = sum(h is not None for h in (results.left_hand_landmarks, results.right_hand_landmarks))
            cv2.putText(view, f"{fps:4.1f} fps  hands: {hands}  label: {args.label}  saved: {saved}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if recording:
                cv2.circle(view, (view.shape[1] - 25, 25), 10, (0, 0, 255), -1)
                cv2.putText(view, f"REC {len(frames)}", (view.shape[1] - 130, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Sign capture (SPACE: record, Q: quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" ") and not args.video:
                if recording and frames:
                    path = save(frames, args.label, args.out)
                    saved += 1
                    print(f"saved {len(frames)} frames -> {path}")
                frames, recording = [], not recording

    if args.video and frames:
        path = save(frames, args.label, args.out)
        print(f"saved {len(frames)} frames -> {path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
