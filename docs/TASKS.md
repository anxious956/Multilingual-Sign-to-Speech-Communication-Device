# Task split (draft)

**Draft for the team to agree on.** Based on the technical goals in the proposal (Section 4, Table 4) and the course schedule in `PROJECT_NOTES.md`. Change names freely; tick items as they are done.

## Technical work

### Goal 1, sign recognition: Asmar Hasanova

Progress and details: `GOAL1_LOG.md`.

- [x] Webcam capture, sign start/end, live demo on a laptop
- [x] First models on the Kaggle ISLR data (v2: 62.3% top-1 on unseen signers, 250 signs)
- [x] v3 (Kaggle 1st-place recipe) and v4 (review fixes, camera-independent input): 67.2% top-1 on 250 Kaggle signs
- [x] Calibration and abstention: confidence threshold, risk–coverage curve (v2, v3, v4)
- [x] ASL Citizen landmarks for the 155 target glosses; v5 = v4 fine-tuned on them: 86.6% top-1 / 98.2% top-5, 134 words, 11 unseen Deaf signers
- [ ] v5 calibration, confusions, live test; "other" class for signs outside the vocabulary
- [ ] Our own test recordings with our camera (team members, then NJIT ASL signers)
- [ ] Vocabulary for the service counter, with NJIT ASL (with the whole team)
- [x] Retrain on ASL Citizen for that vocabulary (v5, draft 155-gloss list; retrain when the final list is chosen)
- [x] Hand-off format to Goal 2 (see "Interfaces" below): `live_demo.py --jsonl`
- [x] Calibration analysis of v2 (`calibrate.py`, results in `GOAL1_LOG.md`)

### Goal 2, language and speech: Dorukhan Cakir and Zeynep Hafsa Cakici

**Dorukhan: signs → English sentence, and the confirmation screen**
- [ ] Templates for common service-counter phrases (e.g. "where store" → "Where is the store?")
- [ ] Small offline language model as a fallback for sign sequences without a template; compare a few models on speed and quality
- [ ] Confirmation screen: show the sentence and the top candidates, let the user confirm or pick another

**Zeynep: translation, speech and the reply channel**
- [ ] NLLB-200 (distilled 600M) offline, English → Turkish and Spanish; check quality on our phrases
- [ ] Piper TTS voices for English, Turkish and Spanish
- [ ] Reply channel: Whisper speech recognition → text on screen; pick the model size that is fast enough
- [ ] Measure each step's time on a laptop (input for the 1.2 s budget)

### Goal 3, offline embedded system: Merrick Simmons

- [ ] Buy and set up the Jetson Orin Nano (JetPack)
- [ ] Check that MediaPipe Holistic runs on the Jetson (ARM); find an alternative early if it does not
- [ ] Run the Goal 1 model with ONNX Runtime / TensorRT; first a speed test with the Kaggle 1st-place model (Hugging Face)
- [ ] Latency budget: confirmation → start of speech under 1.2 s, measured end to end
- [ ] Power draw; decide battery or wall power
- [ ] Camera, microphone, speaker, display; enclosure
- [ ] Program that runs the whole pipeline on the device

## Course deliverables (proposal)

Suggested owners; each person writes the part closest to their goal.

| Week | Deliverable | Suggested owner |
|---|---|---|
| 5 | Constraints: technical | Merrick |
| 5 | Constraints: legal (ADA, interpreter rights, privacy) | Zeynep |
| 5 | Constraints: budget (Jetson and parts cost list) | Merrick |
| 6 | Design methodology | Everyone, one paragraph per goal |
| 7–8 | Literature review, originality, patents | Asmar (done; verify patent legal status) |
| 7–8 | Marketability and comparison with SignAll, Sign-Speak, Avodah | Zeynep |
| 9 | Flow chart (draft in Section 7, Fig. 5) | Dorukhan |
| 9 | Block diagram (draft in Section 7, Fig. 6) | Merrick |
| 9 | Specification table | Asmar |
| — | Environmental and economic factors (power, e-waste, cost vs interpreter) | Merrick and Dorukhan |
| — | Cover page team number | Whoever gets it first |
| — | Contact NJIT ASL community | Asmar and Zeynep |
| 10–12 | Proposal defense slides | Everyone, own goal |
| 12–14 | Proposal report | Everyone, own goal; one person merges |

## Interfaces between goals

Agreeing on these early lets everyone work in parallel with fake data.

- **Goal 1 → Goal 2:** one JSON line per stretch of signing, written by `python src/live_demo.py ... --jsonl out.jsonl`:
  `{"model": "islr_model_v2.pt", "threshold": 0.45, "signs": [{"sign": "water", "prob": 0.86, "top": [["water", 0.86], ["drink", 0.07], ...], "start": 1.6, "end": 3.1, "abstain": false}, ...]}`
  Times are seconds since the demo started; `top` has up to 5 candidates; `abstain` is true when `prob` is below the threshold (ask the user to repeat or pick from `top`). Goal 2 can develop against a saved file of these lines.
- **Goal 2 → speech:** the confirmed English sentence (text) and the chosen output language (`en`, `tr`, `es`).
- **Goal 3:** runs Goals 1 and 2 on the Jetson and measures the time and power of each step.
