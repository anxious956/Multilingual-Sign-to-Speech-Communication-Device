# Notebooks (Goal 1, step 3)

Code to paste into Kaggle or Colab notebooks. Each file is one cell; the comment at its top says where to run it and what it produces.

| File | Runs on | Does |
|---|---|---|
| `01_preprocess_islr.py` | Kaggle notebook created from the [competition page](https://www.kaggle.com/competitions/asl-signs) (the data is attached there) | Shrinks the 56 GB Kaggle ISLR data to 2 GB: 94 landmarks, x/y/z, float16. About 10 minutes. |
| `02_train_islr.py` | Colab (GPU) | v1 / v2 model: 1D-CNN + transformer, 60 epochs, about 15 minutes on an L4. |
| `03_train_islr_v3.py` | Colab (GPU) | v3: the Kaggle 1st-place recipe (late dropout, AWP, stronger augmentation), 300 epochs, about 3 hours on an L4, resumable. |
| `04_train_islr_v4.py` | Colab (GPU) | v4: v3 plus the review fixes (lip mirroring, weight decay) and new input: body coordinates with the camera's aspect ratio removed, each hand relative to its wrist, hand-found flags. Same time as v3. |

## ASL Citizen landmarks (`tools/extract_asl_citizen.py`)

Runs on any Colab runtime (CPU is enough; more CPU cores = faster). Reads the videos straight from Microsoft's zip, so nothing large is downloaded. Measured here: about 65 frames per second with 4 CPU cores, i.e. about 1 hour per 4,000 videos on 4 cores. Stopping and re-running continues where it stopped.

```python
# cell 1
from google.colab import drive
drive.mount("/content/drive")
!git clone -q -b claude/great-hamilton-9vy4ph https://github.com/anxious956/Multilingual-Sign-to-Speech-Communication-Device.git /content/repo
!pip install -q uv
!uv venv /content/mp --python 3.11 -q
!uv pip install -q --python /content/mp mediapipe==0.10.14 "numpy<2" opencv-python-headless remotezip
!nproc
```
```python
# cell 2: first the candidate vocabulary (about 5,000 videos), then everything
OUT = "/content/drive/MyDrive/sign2speech/asl_citizen"
!cd /content/repo && /content/mp/bin/python -W ignore tools/extract_asl_citizen.py --out {OUT} --glosses tools/vocab_glosses_v0.txt --tag vocab0
!cd /content/repo && /content/mp/bin/python -W ignore tools/extract_asl_citizen.py --out {OUT} --merge
```

For all 83,399 videos use `--glosses all --tag all` (about a day on 4 cores; split over several runtimes with `--shard 0 --shards 3`, `--shard 1 --shards 3`, ...). ASL Citizen may not be redistributed: the output stays in the team Drive.

## Data on Google Drive

`MyDrive/sign2speech/` holds the output of step 01 (`islr_landmarks.npy`, `islr_meta.npz`, `islr_info.json`) and the trained models. Getting it there: save the Kaggle notebook's output as a private dataset, download the files one by one and upload them to Drive (the Kaggle API returned 404 for this private output). Copy them to `/content` at the start of every Colab session; reading from Drive during training is slow.

## Signer split

Signers, not signs, are split 15 / 3 / 3 into training, validation and test (seed 42, the same in every version), so the accuracy printed is for people the model has never seen. Validation picks the best epoch; test is measured once at the end.

## Using a trained model

Copy `islr_model_v*.pt` from Drive to `models/` on the laptop and run `src/live_demo.py --model models/islr_model_v3.pt`. `src/sign_model.py` holds copies of the model code and must stay identical to these files.
