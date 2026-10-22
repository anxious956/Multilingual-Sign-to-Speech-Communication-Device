# Notebooks (Goal 1, step 3)

Code to paste into Kaggle or Colab notebooks. Each file is one cell; the comment at its top says where to run it and what it produces.

| File | Runs on | Does |
|---|---|---|
| `01_preprocess_islr.py` | Kaggle notebook created from the [competition page](https://www.kaggle.com/competitions/asl-signs) (the data is attached there) | Shrinks the 56 GB Kaggle ISLR data to 2 GB: 94 landmarks, x/y/z, float16. About 10 minutes. |
| `02_train_islr.py` | Colab (GPU) | v1 / v2 model: 1D-CNN + transformer, 60 epochs, about 15 minutes on an L4. |
| `03_train_islr_v3.py` | Colab (GPU) | v3: the Kaggle 1st-place recipe (late dropout, AWP, stronger augmentation), 300 epochs, about 3 hours on an L4, resumable. |
| `04_train_islr_v4.py` | Colab (GPU) | v4: v3 plus the review fixes (lip mirroring, weight decay) and new input: body coordinates with the camera's aspect ratio removed, each hand relative to its wrist, hand-found flags. Same time as v3. |
| `05_finetune_asl_citizen.py` | Colab (GPU) | v5: v4 fine-tuned on the ASL Citizen landmarks of our 155 target glosses (variants merged into words), official 35 / 6 / 11 signer split, test reported once. Under an hour on an L4. See below. |

## ASL Citizen landmarks (`tools/extract_asl_citizen.py`)

Runs on any Colab runtime (CPU is enough; more CPU cores = faster, and the GPU runtimes come with more cores: `!nproc` shows how many). Reads the videos straight from Microsoft's zip, so nothing large is downloaded. Measured: the 4,755 videos of the 155 target glosses took about an hour with 12 cores (about 5 minutes per chunk of 400 videos). Stopping and re-running continues where it stopped. The script sets `MPLBACKEND=Agg` itself: Colab's own setting is not valid in the separate Python 3.11 environment and made every worker crash on `import mediapipe`.

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

For all 83,399 videos use `--glosses all --tag all` (about 18 hours on 12 cores; split over several runtimes with `--shard 0 --shards 3`, `--shard 1 --shards 3`, ...). ASL Citizen may not be redistributed: the output stays in the team Drive.

## Fine-tuning on ASL Citizen (`05_finetune_asl_citizen.py`)

Needs `islr_model_v4.pt` (from 04) and the merged ASL Citizen files in Drive. Runtime -> **L4 or A100 GPU**.

```python
# cell 1
from google.colab import drive
import os
drive.mount("/content/drive")
D = "/content/drive/MyDrive/sign2speech"
!cp {D}/asl_citizen/asl_citizen_landmarks.npy {D}/asl_citizen/asl_citizen_meta.npz {D}/asl_citizen/asl_citizen_info.json /content/
os.environ["DATA"] = "/content"
os.environ["OUT"] = D
!git clone -q -b claude/great-hamilton-9vy4ph https://github.com/anxious956/Multilingual-Sign-to-Speech-Communication-Device.git /content/repo
```
```python
# cell 2
%run /content/repo/notebooks/05_finetune_asl_citizen.py
```

What it prints first is **v4 without any fine-tuning** on the ASL Citizen test videos of the words that are in both datasets: the domain gap (phone videos -> webcam videos of Deaf signers) that fine-tuning has to close. Then 5 epochs of the output layer alone, then the whole model (100 epochs, stops early after 30 without improvement). At the end: test top-1 / top-5 over the 11 test signers, per signer, and the words also in Kaggle vs the words new to the model. `INIT=scratch` in cell 1 (`os.environ["INIT"] = "scratch"`) trains the same model from random weights, to measure what the Kaggle pretraining is worth; use a different `OUT` folder or delete `asl_citizen_v5_last.pt` and `asl_citizen_model_v5.pt` in between.

Outputs in Drive: `asl_citizen_model_v5.pt` (for `src/live_demo.py --model`, no `--x-scale`), `asl_citizen_v5_logits.npz` and `asl_citizen_v5_info.json`:

```
python src/calibrate.py --logits models/asl_citizen_v5_logits.npz --info models/asl_citizen_v5_info.json --vocab hello,thankyou,... --out docs/figures/calib_v5
python src/confusions.py --logits v5=models/asl_citizen_v5_logits.npz --info models/asl_citizen_v5_info.json
```

(`--meta` is not needed from v4 on: the logits file carries the signer of every output.)

## Data on Google Drive

`MyDrive/sign2speech/` holds the output of step 01 (`islr_landmarks.npy`, `islr_meta.npz`, `islr_info.json`) and the trained models. Getting it there: save the Kaggle notebook's output as a private dataset, download the files one by one and upload them to Drive (the Kaggle API returned 404 for this private output). Copy them to `/content` at the start of every Colab session; reading from Drive during training is slow.

## Signer split

Signers, not signs, are split 15 / 3 / 3 into training, validation and test (seed 42, the same in every version), so the accuracy printed is for people the model has never seen. Validation picks the best epoch; test is measured once at the end.

## Using a trained model

Copy `islr_model_v*.pt` from Drive to `models/` on the laptop and run `src/live_demo.py --model models/islr_model_v3.pt`. `src/sign_model.py` holds copies of the model code and must stay identical to these files.
