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
import json
import time
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from capture_landmarks import draw, frame_landmarks, save
from fluent_segment import best_split, candidate_cuts, merge_repeats
from segment_signs import hand_speed, raised
from sign_model import SignClassifier

REPO = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore", message=".*GetPrototype.*")  # harmless protobuf warning printed on every frame


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
    parser.add_argument("--word-bonus", type=float, default=0.5,
                        help="see fluent_segment.py; larger = more, shorter signs (default: 0.5)")
    parser.add_argument("--pause", type=float, default=0.6,
                        help="see fluent_segment.py; larger = more candidate cuts (default: 0.6)")
    parser.add_argument("--save", action="store_true",
                        help="save every stretch of signing to data/recordings/LIVE to check it later "
                             "with fluent_segment.py")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="below this confidence a sign is not accepted and the user is asked to repeat; 0.45 kept "
                             "the error of accepted signs near 5%% for the 30-sign list with v2 (docs/GOAL1_LOG.md)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature T from calibrate.py for this model and vocabulary (v3 with the 30-sign list: "
                             "0.79); the threshold refers to these calibrated probabilities")
    parser.add_argument("--jsonl", help="append every result to this file, one JSON line per stretch of signing "
                                        "(the Goal 1 -> Goal 2 hand-off format in docs/TASKS.md)")
    parser.add_argument("--rest-line", type=float, default=1.0, help="see segment_signs.py (default: 1.0)")
    parser.add_argument("--min-frames", type=int, default=8, help="shortest sign, in frames (default: 8)")
    parser.add_argument("--rest-frames", type=int, default=6, help="frames at rest that end a sign (default: 6)")
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.x_scale, args.temperature)
    if args.vocab == "list":
        raise SystemExit(" ".join(sorted(classifier.signs)))
    if args.vocab:
        classifier.limit_to([w.strip() for w in args.vocab.split(",")])
    print(f"model loaded: {len(classifier.signs)} signs" + (f", limited to {args.vocab}" if args.vocab else ""))
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera or video")

    sign, rest = [], 0           # frames of the sign in progress, frames at rest since the hands went down
    times, before_t = [], []     # capture time of each frame in `sign` / `before`
    t0 = time.time()
    before = []                  # a few frames before the sign starts, kept as padding
    result = []                  # last prediction: top 3 for one sign
    accepted = True              # whether that sign passed the threshold
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
            now = time.time() - t0
            up = raised(lm[None], args.rest_line)[0]

            if up:
                if not sign:
                    sign, times = before[:], before_t[:]
                sign.append(lm)
                times.append(now)
                rest = 0
            elif sign:
                sign.append(lm)
                times.append(now)
                rest += 1
                if rest >= args.rest_frames:  # hands back at rest: the sign is over
                    if len(sign) - rest >= args.min_frames:
                        frames = np.stack(sign)
                        if args.save:
                            print("saved", save(sign, "LIVE", REPO / "data" / "recordings"))
                        pieces = [] if args.single else merge_repeats(best_split(
                            frames, classifier, candidate_cuts(hand_speed(frames), args.pause),
                            word_bonus=args.word_bonus))
                        if len(pieces) <= 1:
                            pieces = [(0, len(frames), None, None)]
                        out = []
                        for a, b, _, _ in pieces:
                            top = classifier.predict(frames[a:b], top=5)
                            out.append({"sign": top[0][0], "prob": round(top[0][1], 3),
                                        "top": [[s, round(p, 3)] for s, p in top],
                                        "start": round(times[a], 2), "end": round(times[b - 1], 2),
                                        "abstain": top[0][1] < args.threshold})
                        if len(out) > 1:
                            sentence, result = [(o["sign"], o["prob"], o["abstain"]) for o in out], []
                        else:
                            result, sentence = [tuple(x) for x in out[0]["top"][:3]], []
                            accepted = not out[0]["abstain"]
                        print(" | ".join(f"{o['sign']} {o['prob']:.0%}" + (" (not sure)" if o["abstain"] else "")
                                         for o in out))
                        if args.jsonl:
                            with open(args.jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"model": Path(args.model).name, "threshold": args.threshold,
                                                    "signs": out}) + "\n")
                    sign, times, rest = [], [], 0
            before, before_t = (before + [lm])[-3:], (before_t + [now])[-3:]

            draw(frame, results)
            view = cv2.flip(frame, 1)
            status = "signing..." if sign else "ready"
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if result and not accepted:
                cv2.putText(view, "not sure - please repeat", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            for k, (s, p) in enumerate(result):
                first = k == 0 and accepted
                cv2.putText(view, f"{s}  {p:.0%}", (10, 110 + 35 * k if not accepted else 70 + 35 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0 if first else 0.7, (0, 255, 0) if first else (200, 200, 200), 2)
            if sentence:
                x = 10
                for s, p, unsure in sentence:  # signs below the threshold in orange with "?"
                    word = f"{s}?" if unsure else s
                    cv2.putText(view, word, (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 165, 255) if unsure else (0, 255, 0), 2)
                    cv2.putText(view, f"{p:.0%}", (x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    x += cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] + 25
            cv2.imshow("Live sign recognition (Q: quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
