"""Live sign recognition from the webcam (Goal 1, step 4).

The webcam runs MediaPipe Holistic as in capture_landmarks.py. A sign starts
when a hand is raised and ends when the hands are back at rest
(segment_signs.py); the frames in between go to the trained model and the top
three guesses with their probabilities are shown on the screen.

Usage:
    python src/live_demo.py --model models/islr_model.pt

Keys: Q quits.
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from capture_landmarks import draw, frame_landmarks
from fluent_segment import best_split, candidate_cuts, merge_repeats
from segment_signs import hand_speed, raised
from sign_model import SignClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/islr_model.pt", help="trained model (islr_model.pt from Colab)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--video", help="run on a video file instead of the webcam")
    parser.add_argument("--x-scale", type=float, default=2.08,
                        help="stretch x to the Kaggle videos' proportions; 2.08 measured for a 4:3 laptop webcam "
                             "with check_aspect.py (Kaggle 3.25 / ours 1.56); 1 = no change")
    parser.add_argument("--vocab", help="only these signs, comma separated, e.g. hello,bye,yes,no; "
                                        "--vocab list prints all 250")
    parser.add_argument("--single", action="store_true",
                        help="one sign per raise of the hands (default: several signs may be made without "
                             "lowering the hands; they are split with fluent_segment.py)")
    parser.add_argument("--rest-line", type=float, default=1.0, help="see segment_signs.py (default: 1.0)")
    parser.add_argument("--min-frames", type=int, default=8, help="shortest sign, in frames (default: 8)")
    parser.add_argument("--rest-frames", type=int, default=6, help="frames at rest that end a sign (default: 6)")
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.x_scale)
    if args.vocab == "list":
        raise SystemExit(" ".join(sorted(classifier.signs)))
    if args.vocab:
        classifier.limit_to([w.strip() for w in args.vocab.split(",")])
    print(f"model loaded: {len(classifier.signs)} signs" + (f", limited to {args.vocab}" if args.vocab else ""))
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera or video")

    sign, rest = [], 0           # frames of the sign in progress, frames at rest since the hands went down
    before = []                  # a few frames before the sign starts, kept as padding
    result = []                  # last prediction: top 3 for one sign
    sentence = []                # last prediction: signs in order when several were made in one go
    failed = 0
    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed += 1
                if args.video or failed > 50:  # end of the video, or the camera gives no image
                    print("no more frames" if args.video else
                          "The camera gives no image. Close other apps using it (Zoom, Teams, browser) or try --camera 1.")
                    break
                time.sleep(0.1)
                continue
            failed = 0
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = frame_landmarks(results)
            up = raised(lm[None], args.rest_line)[0]

            if up:
                if not sign:
                    sign = before[:]
                sign.append(lm)
                rest = 0
            elif sign:
                sign.append(lm)
                rest += 1
                if rest >= args.rest_frames:  # hands back at rest: the sign is over
                    if len(sign) - rest >= args.min_frames:
                        frames = np.stack(sign)
                        pieces = [] if args.single else merge_repeats(
                            best_split(frames, classifier, candidate_cuts(hand_speed(frames))))
                        if len(pieces) > 1:
                            sentence, result = [(s, p) for _, _, s, p in pieces], []
                            print(" | ".join(f"{s} {p:.0%}" for s, p in sentence))
                        else:
                            result, sentence = classifier.predict(frames), []
                            print("  ".join(f"{s} {p:.0%}" for s, p in result))
                    sign, rest = [], 0
            before = (before + [lm])[-3:]

            draw(frame, results)
            view = cv2.flip(frame, 1)
            status = "signing..." if sign else "ready"
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for k, (s, p) in enumerate(result):
                cv2.putText(view, f"{s}  {p:.0%}", (10, 70 + 35 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0 if k == 0 else 0.7, (0, 255, 0) if k == 0 else (200, 200, 200), 2)
            if sentence:
                cv2.putText(view, "  ".join(s for s, _ in sentence), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)
                cv2.putText(view, "  ".join(f"{p:.0%}" for _, p in sentence), (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("Live sign recognition (Q: quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
