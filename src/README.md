# Source code

## Goal 1, step 1: record signs as landmarks

`capture_landmarks.py` opens the webcam, runs MediaPipe Holistic, draws the detected face, body and hand points, and saves each recorded sign as a NumPy file of shape `(frames, 543, 3)`. The landmark order matches the Google Isolated Sign Language Recognition (Kaggle) dataset, so our recordings and that dataset can be used with the same model. Landmarks that are not detected are stored as NaN.

### Setup (Windows)

Use Python 3.9–3.12; MediaPipe 0.10.14 does not support Python 3.13. Check with `python --version`. If you have several versions installed, `py -0` lists them and `py -3.12` picks one.

Put everything on drive D: (repo, virtual environment, pip downloads and recordings):

```powershell
D:
cd \
git clone https://github.com/anxious956/Multilingual-Sign-to-Speech-Communication-Device.git
cd Multilingual-Sign-to-Speech-Communication-Device
git checkout claude/great-hamilton-9vy4ph
py -3.12 -m venv .venv
.venv\Scripts\activate
$env:PIP_CACHE_DIR = "D:\pip-cache"
pip install -r requirements.txt
```

Without Git: on GitHub choose the branch `claude/great-hamilton-9vy4ph`, then **Code → Download ZIP**, and extract it to `D:\`.

Next time, only open PowerShell and run:

```powershell
cd D:\Multilingual-Sign-to-Speech-Communication-Device
.venv\Scripts\activate
```

### Run

```powershell
python src\capture_landmarks.py --label HELLO
```

- **SPACE** starts recording; press it again to stop and save the sign.
- **Q** quits.
- The top line shows frames per second, how many hands are detected, the label and how many signs were saved.

Recordings are saved inside the repo folder, e.g. `D:\Multilingual-Sign-to-Speech-Communication-Device\data\recordings\<LABEL>\<LABEL>_<date>_<time>.npy`, from wherever the script is run. Use `--out D:\some\folder` to save elsewhere. The `data/` folder is not committed to Git.

To process an existing video file instead of the webcam:

```powershell
python src\capture_landmarks.py --video clip.mp4 --label HELLO
```

### Tips

- Sit so that your head, shoulders and both hands are in the frame. Holistic finds the body first and locates the hands from it, so hands alone are not detected.
- Good, even lighting helps. Avoid a bright window behind you.
- If the wrong camera opens, try `--camera 1`.

### Play back recordings

```powershell
python src\view_recording.py data\recordings\HELLO
```

Shows each recording as a moving skeleton (orange = left hand, blue = right hand) and prints how many frames had a hand detected. **SPACE** pauses, **N** goes to the next recording, **Q** quits.

## Goal 1, step 2: find where each sign starts and ends

`segment_signs.py` marks the frames where at least one hand is raised above a rest line below the shoulders; lowered hands (or hands out of the frame) separate signs. Short flickers are smoothed out.

```powershell
python src\segment_signs.py data\recordings\HELLO
```

For each recording it prints the signs found as frame ranges and a timeline (`#` = hand raised, `.` = at rest). Add `--trim` to save each sign on its own to `data\segments\<LABEL>\`. If signs are cut too early or a resting hand counts as raised, change `--rest-line` (default 1.0 shoulder width below the shoulders; larger = lower line).
