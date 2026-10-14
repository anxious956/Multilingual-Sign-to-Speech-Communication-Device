# Multilingual Sign-to-Speech Communication Device

Edge-deployed American Sign Language recognition with confidence-gated multilingual speech output.

NJIT ECE 414 — Senior Design Project I

---

## What it does

A portable device that lets a Deaf user sign to a camera and have the result spoken aloud in a selected language, while the hearing person's spoken reply appears back as on-screen text.

The system **abstains instead of guessing**: a sentence is only voiced after the user confirms it on screen. Low-confidence readings trigger a candidate list rather than a wrong utterance.

## Pipeline

```
Camera → MediaPipe Holistic (94 of its 543 landmarks: lips, hands, upper body)
       → sign start/end from the rest pose; several signs in a row split with the model's help
       → ISLR model (isolated sign recognition, top-k + confidence)
       → gloss lattice → natural English sentence
       → user confirmation
       → NLLB-200 (EN → target language)
       → Piper TTS → speaker

Microphone → Whisper → on-screen text (reverse channel)
```

## Technical goals

| Goal | What | Owner |
|---|---|---|
| 1. Sign recognition | Webcam → landmarks → signs with a calibrated confidence | Asmar Hasanova |
| 2. Language and speech | Signs → English sentence, translation (NLLB-200), speech (Piper), reply channel (Whisper) | Dorukhan Cakir, Zeynep Hafsa Cakici |
| 3. Offline embedded system | Everything running on the Jetson Orin Nano, latency and power | Merrick Simmons |

Project-wide tasks (vocabulary with Deaf ASL users, testing, report) are shared by the whole team.

## Target specifications

| Metric | Target |
|---|---|
| Vocabulary | 100–250 signs + fingerspelling fallback |
| Sign accuracy (signer-independent) | ≥75% top-1, ≥90% top-5 |
| Sign → speech latency | < 1.2 s |
| Output languages | English, Turkish, Spanish |
| Connectivity | Fully offline |

*These are design targets, not measured results. Measured values will be published here as the project progresses.*

## Hardware

- Embedded GPU platform (NVIDIA Jetson Orin Nano)
- RGB camera
- USB microphone
- Speaker
- Display

## Stack

Python · PyTorch · MediaPipe · ONNX Runtime / TensorRT · NLLB-200 · Piper TTS · Whisper

## Datasets

- [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) (Kaggle) — first models: 94,477 signs, 250 classes, 21 signers, already MediaPipe landmarks
- [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) — primary for the final vocabulary, consent-based
- [WLASL](https://dxli94.github.io/WLASL/) — benchmarking
- Google ASL Fingerspelling (Kaggle) — fingerspelling module

## Status

🚧 In development (Fall 2026). Proposal done (`docs/`). Goal 1 (sign recognition) has a working laptop prototype: webcam capture, sign segmentation, a model trained on the Kaggle data and a live demo.

| Model (Kaggle ISLR, 250 signs) | Test top-1 | Test top-5 |
|---|---|---|
| v2 | 62.3% | 79.5% |

Test = 3 signers never seen in training. v3 (the Kaggle 1st-place recipe) is training. Details: [`docs/GOAL1_LOG.md`](docs/GOAL1_LOG.md).

## Quick start: live demo on a laptop (Windows)

Needs Python 3.9–3.12 and a webcam. Full instructions: [`src/README.md`](src/README.md).

```cmd
git clone https://github.com/anxious956/Multilingual-Sign-to-Speech-Communication-Device.git
cd Multilingual-Sign-to-Speech-Communication-Device
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python src\live_demo.py --model models\islr_model_v2.pt --vocab hello,bye,please,thankyou,yes,no,water
```

The trained model (`islr_model_v2.pt`) is not in Git; copy it from the team Drive folder `sign2speech` to `models/`. Raise a hand to sign, lower it when done; several signs can be made without lowering the hands in between.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Laptop scripts: capture, segmentation, live demo ([setup](src/README.md)) |
| `notebooks/` | Data preparation (Kaggle) and training (Colab) ([how to run](notebooks/README.md)) |
| `docs/` | Proposal ([PDF](docs/ECE_414_Proposal_Draft.pdf)), [project notes and decisions](docs/PROJECT_NOTES.md), [Goal 1 log](docs/GOAL1_LOG.md) |
| `literature/` | Papers, patents and their summaries ([overview](literature/README.md)) |
| `lecture-notes/` | ECE 414 course material |

## Team

Zeynep Hafsa Cakici · Dorukhan Cakir · Merrick Simmons · Asmar Hasanova

## Note on scope

This system recognises signs from a fixed vocabulary, one at a time or a few in a row, and produces word-level output. It does **not** translate continuous ASL and does not model non-manual grammatical markers such as facial expression. It is an assistive communication aid, not a replacement for a qualified human interpreter.
