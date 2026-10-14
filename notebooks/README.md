# Notebooks (Goal 1, step 3)

Code to paste into Kaggle or Colab notebooks. Each file is one cell; the comment at its top says where to run it and what it produces.

| File | Runs on | Does |
|---|---|---|
| `01_preprocess_islr.py` | Kaggle notebook created from the [competition page](https://www.kaggle.com/competitions/asl-signs) (the data is attached there) | Shrinks the 56 GB Kaggle ISLR data to 2 GB: 94 landmarks, x/y/z, float16. About 10 minutes. |
| `02_train_islr.py` | Colab (GPU) | v1 / v2 model: 1D-CNN + transformer, 60 epochs, about 15 minutes on an L4. |
| `03_train_islr_v3.py` | Colab (GPU) | v3: the Kaggle 1st-place recipe (late dropout, AWP, stronger augmentation), 300 epochs, about 3 hours on an L4, resumable. |

## Data on Google Drive

`MyDrive/sign2speech/` holds the output of step 01 (`islr_landmarks.npy`, `islr_meta.npz`, `islr_info.json`) and the trained models. Getting it there: save the Kaggle notebook's output as a private dataset, download the files one by one and upload them to Drive (the Kaggle API returned 404 for this private output). Copy them to `/content` at the start of every Colab session; reading from Drive during training is slow.

## Signer split

Signers, not signs, are split 15 / 3 / 3 into training, validation and test (seed 42, the same in every version), so the accuracy printed is for people the model has never seen. Validation picks the best epoch; test is measured once at the end.

## Using a trained model

Copy `islr_model_v*.pt` from Drive to `models/` on the laptop and run `src/live_demo.py --model models/islr_model_v3.pt`. `src/sign_model.py` holds copies of the model code and must stay identical to these files.
