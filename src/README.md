# Source code

## Goal 1, step 1: record signs as landmarks

`capture_landmarks.py` opens the webcam, runs MediaPipe Holistic, draws the detected face, body and hand points, and saves each recorded sign as a NumPy file of shape `(frames, 543, 3)`. The landmark order matches the Google Isolated Sign Language Recognition (Kaggle) dataset, so our recordings and that dataset can be used with the same model. Landmarks that are not detected are stored as NaN.

### Setup (Windows)

Use Python 3.9–3.12; MediaPipe 0.10.14 does not support Python 3.13. Check with `python --version`. If you have several versions installed, `py -0` lists them and `py -3.12` picks one.

```powershell
cd Multilingual-Sign-to-Speech-Communication-Device
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```powershell
python src\capture_landmarks.py --label HELLO
```

- **SPACE** starts recording; press it again to stop and save the sign.
- **Q** quits.
- The top line shows frames per second, how many hands are detected, the label and how many signs were saved.

Recordings are saved to `data\recordings\<LABEL>\<LABEL>_<date>_<time>.npy`. The `data/` folder is not committed to Git.

To process an existing video file instead of the webcam:

```powershell
python src\capture_landmarks.py --video clip.mp4 --label HELLO
```

### Tips

- Sit so that your head, shoulders and both hands are in the frame. Holistic finds the body first and locates the hands from it, so hands alone are not detected.
- Good, even lighting helps. Avoid a bright window behind you.
- If the wrong camera opens, try `--camera 1`.
