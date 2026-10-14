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
py -3.12 -m venv .venv
.venv\Scripts\activate
$env:PIP_CACHE_DIR = "D:\pip-cache"
pip install -r requirements.txt
```

Without Git: on GitHub choose **Code → Download ZIP**, and extract it to `D:\`.

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

## Goal 1, step 4: live recognition

`live_demo.py` runs the webcam, finds each sign with the rest-pose rule from step 2 and shows the model's top three guesses with probabilities. It needs a trained model from Colab (`islr_model_v2.pt` or `islr_model_v3.pt` in the Drive `sign2speech` folder) copied to `models\`, and PyTorch (CPU version is enough):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
python src\live_demo.py --model models\islr_model_v2.pt
```

`sign_model.py` holds the model and its input preparation; it must stay identical to `notebooks/02_train_islr.py`.

## Several signs without lowering the hands

`fluent_segment.py` splits a stretch of signing with the hands up into separate signs: places where the hands slow down are candidate cuts, and the sign model picks the cuts whose pieces it recognises most confidently. The live demo uses it by default (`--single` turns it off).

```powershell
python src\fluent_segment.py data\recordings\FLUENT --model models\islr_model_v2.pt --vocab hello,water,please
```

`segment_signs.py --pauses` is the speed-only rule; it cuts signs in half where the hands stop inside a sign, so it is kept only for comparison.

## Goal 1, step 5: calibration and abstention

`calibrate.py` checks whether the model's confidence can be trusted, using the validation and test outputs the training notebooks save (no training needed). Temperature and thresholds are fitted on the validation signers and reported on the test signers: accuracy per signer, calibration error before and after temperature scaling, the risk–coverage curve, the confidence threshold that keeps the error of answered signs at 5% or 10%, a candidate list that contains the right sign 90% of the time, and, with `--vocab`, how often signs outside the vocabulary would still be accepted.

Copy `islr_val_logits.npz` (v2) or `islr_v3_logits.npz` (v3), `islr_meta.npz` and `islr_info.json` from the Drive `sign2speech` folder to `models\`, then:

```powershell
python src\calibrate.py --logits models\islr_val_logits.npz --meta models\islr_meta.npz --info models\islr_info.json --vocab hello,bye,please,thankyou,yes,no,water --out docs\figures\calib_v2
```

It writes `docs\figures\calib_v2_report.md` and the reliability and risk–coverage figures.
