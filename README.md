# Multilingual Sign-to-Speech Communication Device

Edge-deployed American Sign Language recognition with confidence-gated multilingual speech output.

NJIT ECE 414 — Senior Design Project I

---

## What it does

A portable device that lets a Deaf user sign to a camera and have the result spoken aloud in a selected language, while the hearing person's spoken reply appears back as on-screen text.

The system **abstains instead of guessing**: a sentence is only voiced after the user confirms it on screen. Low-confidence readings trigger a candidate list rather than a wrong utterance.

## Pipeline

```
Camera → MediaPipe Holistic (130 landmarks)
       → ISLR model (isolated sign recognition, top-k + confidence)
       → gloss lattice → natural English sentence
       → user confirmation
       → NLLB-200 (EN → target language)
       → Piper TTS → speaker

Microphone → Whisper → on-screen text (reverse channel)
```

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

- [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) — primary, consent-based
- [WLASL](https://dxli94.github.io/WLASL/) — benchmarking
- Google ASL Fingerspelling (Kaggle) — fingerspelling module

## Status

🚧 In development — proposal phase (Fall 2026)

## Team

Zeynep Hafsa Cakici · Dorukhan Cakir · Merrick Simmons · Asmar Hasanova

## Note on scope

This system performs isolated-sign recognition and produces word-level output. It does **not** translate continuous ASL and does not model non-manual grammatical markers such as facial expression. It is an assistive communication aid, not a replacement for a qualified human interpreter.
