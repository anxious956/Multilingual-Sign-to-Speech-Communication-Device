"""Compare the body proportions in our recordings with those in the Kaggle data.

MediaPipe gives x as a fraction of the image width and y as a fraction of
the height, so the same body looks wider or narrower depending on the
camera's aspect ratio (a 4:3 laptop webcam vs a 9:16 phone held upright).
This prints shoulder width (x) divided by the shoulder-to-mouth height (y);
run the same measure on the Kaggle data in Colab and compare.

Usage:
    python src/check_aspect.py data/recordings
"""
import sys
from pathlib import Path

import numpy as np

LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
L_SHOULDER, R_SHOULDER = 489 + 11, 489 + 12

ratios = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "data/recordings").rglob("*.npy")):
    a = np.load(f)
    width = np.abs(a[:, L_SHOULDER, 0] - a[:, R_SHOULDER, 0])
    height = (a[:, L_SHOULDER, 1] + a[:, R_SHOULDER, 1]) / 2 - np.nanmean(a[:, LIPS, 1], axis=1)
    ratios.append(np.nanmedian(width / height))
print(f"{len(ratios)} recordings, shoulder width / shoulder-to-mouth height: median {np.nanmedian(ratios):.2f}")
