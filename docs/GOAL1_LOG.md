# Goal 1 (sign recognition) log

Owner: Asmar Hasanova. Code is in `src/` (runs on the laptop) and `notebooks/` (runs on Kaggle and Colab). Setup and usage are in `src/README.md`.

## 13 Oct 2026

### Done

| Step | What | Result |
|---|---|---|
| 1. Webcam capture | `src/capture_landmarks.py` records signs as MediaPipe Holistic landmarks, (frames, 543, 3), in the Kaggle ISLR order. `src/view_recording.py` plays them back as a skeleton. | About 20 frames per second on Asmar's Windows laptop, hands and face detected. |
| 2. Sign start and end | `src/segment_signs.py`: a sign lasts while a hand is above a rest line below the shoulders. | All 5 single-sign recordings and a 3-sign recording (240 frames) were split correctly. |
| 3a. Shrink the Kaggle data | `notebooks/01_preprocess_islr.py`, run in a Kaggle notebook (the data is attached there, nothing downloaded). Keeps 94 landmarks: lips 40, both hands, upper body 12. | 56 GB → 2.02 GB. 94,477 signs, 250 classes, 21 signers. Only 61% of frames have a hand. Saved as the private Kaggle dataset `asmar05/islr-small` and in Google Drive `MyDrive/sign2speech`. |
| 3b. Train | `notebooks/02_train_islr.py` on Colab (L4 GPU). 1D-CNN + transformer, 1.65 M parameters. Signers split 15 / 3 / 3 (train / validation / test, about 70/15/15), so test signers are never seen. | See the table below. |
| 4. Live demo | `src/live_demo.py` + `src/sign_model.py`: webcam → step 2 → model → top 3 guesses on screen. | Works end to end on the laptop. |

### Model results (test = 3 signers never seen)

| Model | Change | Val top-1 | Test top-1 | Test top-5 | Train acc at end |
|---|---|---|---|---|---|
| v1 | 40 epochs, mirror / rotate / stretch augmentation | 70.9% | 61.0% | 77.6% | 99.1% |
| v2 | 60 epochs, plus speed change, dropped frames / hands / lips, more dropout | 70.9% | 62.3% | 79.5% | 94.4% |
| **v3** | Kaggle 1st-place recipe: late dropout + AWP, stronger augmentation, 300 epochs (best epoch 272) | **75.8%** | **66.4%** | **83.0%** | 79.5% |

| **v4** | v3 + review fixes (lip mirroring, weight decay), body coordinates without camera aspect, hand-local features, hand-found flags (best epoch 258) | 75.8% | **67.2%** | **83.9%** | 83.8% |

v4 per test signer: 46.2% / 71.1% / 78.7%. On Kaggle v4 is only slightly better than v3 (+0.8 top-1, +0.9 top-5, worst signer +1.8); it led by 1–2 points until about epoch 150, then both reached the same validation ceiling. Its main purpose, independence from the camera's aspect ratio and better hand-shape input, cannot show on Kaggle data (one camera type) and has to be checked live and on our own recordings.

v3 per test signer (4718 / 28656 / 55372): 44.4% / 71.4% / 77.8% (v2: 41.2% / 65.4% / 74.8%). v3 never overfit (train accuracy stayed below validation until about epoch 150) and kept improving until epoch 272, so stopping early at 60 epochs without improvement would have lost about 3 points.

Both models overfit (train accuracy far above validation). v2 overfits less and is slightly better on test. Files: `islr_model_v1.pt`, `islr_model_v2.pt` in Drive `sign2speech`, and copied to `models/` on the laptop (not in Git). Each run also saves validation and test model outputs (`islr_val_logits.npz`) for calibration.

### Problems found and fixed

- **Camera proportions.** MediaPipe x is a fraction of the image width, y of the height. Shoulder width / shoulder-to-mouth height is 3.25 in the Kaggle data and 1.56 on our 4:3 webcam (`src/check_aspect.py`), so the Kaggle videos were taller than wide (phones). The live demo now multiplies x by 2.08 (`--x-scale`). Before this, a correct HELLO got 4%; after it, correct HELLOs got 45–92%.
- **HELLO vs BYE.** In the Kaggle data HELLO is the salute from the forehead; waving is BYE. The model follows that. Decide with the NJIT ASL club whether a wave should count as a greeting.
- **HELLO vs SUN.** Both are near the head; the model often mixes them (e.g. sun 27%, hello 21%). Expected to improve with hand-shape features and a smaller vocabulary.
- Kaggle notebook output could not be downloaded with the Kaggle API or kagglehub (404). Workaround: notebook output → private dataset → download each file by hand → upload to Drive. The browser saved `islr_meta.npz` as `.zip`; it only needed renaming.

### Vocabulary

The Kaggle 250 signs are for parents of young Deaf children (animals, toys…). `--vocab` limits the demo to a chosen list. Current test list (30 signs, enough for short sentences):

hello, bye, please, thankyou, yes, no, fine, minemy, yourself, where, who, why, now, later, tomorrow, wait, go, have, like, drink, water, food, hungry, thirsty, sick, home, store, callonphone, finish, time

Missing for a service counter: want, need, help, pay, how much… These come with ASL Citizen (step 7). The final list is decided with NJIT ASL (step 6).

## 14 Oct 2026: several signs without lowering the hands

Fluent signing has little or no pause between signs (Moryossef et al. 2023), so the rest-pose rule of step 2 sees several signs as one.

**What Moryossef et al. 2023 do.** MediaPipe Holistic poses, normalised to shoulder width 1; a bidirectional LSTM (4 layers, 256) tags every frame B (beginning), I (inside) or O (outside) a sign; a segment starts where P(B) passes a threshold. BIO instead of in/out is what lets two signs touch. Hand speed mainly marks phrase boundaries, and a change of hand shape often marks a sign boundary (only 17% of neighbouring signs share a hand shape). Trained on German Sign Language (DGS); code and models are public (github.com/sign-language-processing/transcription).

**What we tried** on 3 recordings by Asmar of HELLO, WATER, PLEASE without lowering the hands (`data/recordings/FLUENT`):

| Method | Result |
|---|---|
| Hand speed only (`segment_signs.py --pauses`): cut where the hands nearly stop | 3–4 pieces per recording, mostly wrong. The hands also stop inside a sign (HELLO at the forehead, WATER at the chin), so signs were cut in half and real boundaries missed. |
| **Speed + sign model** (`fluent_segment.py`): slow-downs are only candidate cuts; dynamic programming keeps the cuts whose pieces the model recognises most confidently; neighbouring pieces with the same sign are merged | **hello · water · please in all 3 recordings** (30-sign vocabulary). |

Only 3 recordings by one person, so not yet a result to report. A deliberate repetition ("no no") would be merged into one sign.

The live demo now does this by default: make several signs, lower the hands, and the signs appear in order (`--single` for the old one-sign behaviour).

## 14 Oct 2026: calibration of v2 (`src/calibrate.py`)

Fitted on the 3 validation signers, reported on the 3 test signers (12,888 test signs). The 30-sign vocabulary is the demo list.

| | All 250 signs | 30-sign vocabulary |
|---|---|---|
| Top-1 / top-5 (all test signs) | 62.3% / 79.5% | 81.5% / 90.5% |
| Top-1 per test signer (4718 / 28656 / 55372) | 41.2% / 65.4% / 74.8% | 60.2% / 85.5% / 93.3% |
| Temperature T; ECE raw → scaled | 0.98; 6.1% → 7.2% | 0.91; 3.0% → 2.1% |
| Threshold for 5% error (chosen on validation) → test coverage, test error | 0.94 → 19.7%, 5.4% | 0.45 → 82.1%, 6.4% |
| Same threshold, worst signer (4718): coverage, error | 12.3%, 8.1% | 61.2%, 12.2% |
| Split-conformal list, 90% target → test coverage | 83.3% (mean size 7.3) | 81.9% |

Out-of-vocabulary: for the 30-sign list, AUROC 0.975 between correct in-vocabulary signs and the other 220 classes, but at the threshold that keeps 95% of correct in-vocabulary signs, 14.7% of outside signs are still accepted.

**v3 and v4, same analysis** (`docs/figures/calib_v3_report.md`, `calib_v4_report.md`; v4 added 21 Oct 2026):

| | v2, 250 signs | v3, 250 signs | v4, 250 signs | v2, 30 signs | v3, 30 signs | v4, 30 signs |
|---|---|---|---|---|---|---|
| Top-1 / top-5 | 62.3 / 79.5 | 66.4 / 83.0 | 67.2 / 83.9 | 81.5 / 90.5 | 84.0 / 93.0 | 84.4 / 92.8 |
| Worst test signer top-1 | 41.2 | 44.4 | 46.2 | 60.2 | 64.3 | 65.9 |
| T; ECE raw → scaled | 0.98; 6.1 → 7.2 | 0.76; 13.3 → 1.1 | 0.85; 5.1 → 2.5 | 0.91; 3.0 → 2.1 | 0.79; 6.3 → 2.0 | 0.84; 4.4 → 2.4 |
| Coverage at the 5% threshold (test error) | 19.7% (5.4%) | 44.9% (5.1%) | 45.9% (5.1%) | 82.1% (6.4%) | 86.4% (6.4%) | 87.7% (7.2%) |
| AURC | 0.151 | 0.104 | 0.101 | 0.029 | 0.022 | 0.020 |
| Outside-vocabulary signs accepted (95% of correct kept) | — | — | — | 14.7% | 27.7% | 21.7% |

- **v4 is a little better than v3 on every line** (accuracy, worst signer, AURC) and its raw probabilities are closer to right (T 0.85 instead of 0.76, ECE 5% raw instead of 13%). Live demo settings for the 30-sign list: `--temperature 0.84 --threshold 0.47`, and no `--x-scale` (v4 removes the aspect ratio itself). The weakest test signer is still far behind (46% on 250 signs, 66% on 30), and at the 5% threshold that signer still gets 14% error: the signer gap is not something v4's input changes fix.
- Outside-vocabulary acceptance: 21.7%, between v2 and v3. Still needs the "other" class.

- v3 answers far more signs at the same error (250 signs: 45% instead of 20% at 5% error).
- **v3's raw probabilities are too low** (T < 1, from label smoothing and late dropout): ECE 13% raw, 1–2% after temperature scaling. With v3 the live demo must use the fitted temperature (`--temperature 0.79` for the 30-sign list), otherwise the threshold means something else.
- v3 is worse at rejecting signs outside the vocabulary (28% accepted vs 15%), so an "other" class or negative examples matter more.

**Confusions** (`src/confusions.py`, 30-sign list, test signers):
- THANKYOU and FOOD are never confused in the Kaggle test data (0% both ways, v2 and v3), although v3 often read Asmar's THANKYOU as FOOD live. So that confusion comes from our camera or signing style, not from the model in general: our own test recordings are needed.
- Weakest signs with v3: PLEASE 62%, GO 66%, TOMORROW 73%, LIKE 73%. PLEASE ↔ MINE/MY (both on the chest) and WAIT → THANKYOU recur in both models.
- **v3 has a "sink" class: TIME** is the wrong answer for LATER (17%), GO, WHERE, STORE, THIRSTY, YES and WHO. v2 did not do this. Worth checking in v4.
- **v4 (21 Oct 2026):** THANKYOU ↔ FOOD still 2% / 2% on test, so the live confusion is not the model's. v3's weakest signs improved: PLEASE 62 → 70%, GO 66 → 79%, WAIT 76 → 84%, LATER 74 → 77%, LIKE 73 → 76%. The TIME sink is smaller (4 signs instead of 7: LATER 13%, STORE 11%, THIRSTY 8%, WHERE 6%). One sign got worse: MINE/MY 81 → 75% (MINE/MY → PLEASE 15% is now the most frequent confusion); new: WHY → HOME 12%.

What this shows (v2):
- **The raw confidence of v2 is already close to calibrated on average** (ECE 3–6%; temperature scaling changes little).
- **Signers differ a lot.** One test signer gets 41% where another gets 75%. A threshold chosen on other signers gives that signer about twice the intended error. This is the "works worse for some users" concern of Tran et al. 2023, measured.
- **Guarantees fitted on one group of signers do not carry over to new signers**: the conformal list aims at 90% and reaches 82–83%; the 5% threshold gives 5.4–6.4%. The threshold has to be calibrated on data from many signers, ideally from our own camera.
- **A usable operating point exists for the demo vocabulary**: answer about 82% of signs at about 6% error, ask to repeat the rest.
- **Signs outside the vocabulary are a real risk** (15% accepted): needs an "other" class or negative examples.

## Dataset roles (decided 14 Oct 2026, after the second review)

| Dataset | Role | Why | Licence / ethics |
|---|---|---|---|
| Kaggle ISLR (PopSign subset) | Pretraining and fast experiments only; reported final results do not come from it | Child vocabulary, 21 signers, phone portrait video, landmarks only (tied to MediaPipe) | Competition rules: no redistribution |
| PopSign ASL v1.0 | Optional: more signers (47) for the Kaggle signs, plus labelled variants | Same vocabulary as Kaggle; videos, 1.1 TB | CC BY 4.0, consent-based |
| **ASL Citizen** | **Main dataset: target vocabulary, fine-tuning and reported results** | 52 signers (35 train / 6 val / 11 test), webcam at home, 2,731 signs, about 30 videos from about 29 signers per sign, videos (any keypoint model) | Microsoft Research licence: non-commercial research only, **data and derived data may not be redistributed** (keep landmarks and models in the team Drive, not in this public repo) |
| WLASL | Dropped (cited only) | Scraped without consent, dead links, noisy labels; ASL Citizen paper: on shared signs a WLASL-trained model reaches 8.5% top-1 vs 74.2% for ASL Citizen | — |
| Our own recordings | Device test set, calibration (temperature, threshold) for our camera, light fine-tuning | Team recordings only for debugging; reported accuracy from Deaf / fluent NJIT signers | Consent form; ask whether NJIT IRB approval is needed; add timestamps |

ASL Citizen's split files and single videos can be read straight from Microsoft's zip without downloading the 42.8 GB (`remotezip`). `tools/vocab_candidates.py` uses this to compare a draft service-counter list with ASL Citizen and the Kaggle 250: `docs/VOCAB_CANDIDATES.md` (38 words in both, 97 only in ASL Citizen, 6 in neither: how much, return, receipt, restroom, emergency, question; some may exist under other glosses).

**ASL Citizen landmarks extracted (target vocabulary).** `tools/extract_asl_citizen.py --glosses tools/vocab_glosses_v0.txt` on a 12-core Colab runtime: 4,755 videos of the 155 target glosses, 12 chunks, merged into `asl_citizen_landmarks.npy` (207 MB, about 367k frames, same 94 landmarks as Kaggle), `asl_citizen_meta.npz` and `asl_citizen_info.json` in the team Drive (`sign2speech/asl_citizen`), not in this repo (licence). On Colab the venv needs `MPLBACKEND=Agg` (now set in the script). The 12 chunks took about an hour on a 12-core runtime.

## 22 Oct 2026: v5 = v4 fine-tuned on ASL Citizen

`notebooks/05_finetune_asl_citizen.py`, L4, 100 epochs, about 5 minutes. Test = ASL Citizen's official test split: **11 Deaf signers never seen in training or validation**, webcam videos.

| | Value |
|---|---|
| Data | 4,755 videos, 155 glosses → **134 words** (variants merged), 50 signers; train 2,266 videos / 34 signers, validation 576 / 5, test 1,913 / 11 |
| v4 unchanged, ASL Citizen test videos of the 38 words also in Kaggle (choice among those 38) | 88.2% top-1 |
| v5, same videos, same 38-way choice | **97.3%** |
| v5, all 134 words: best validation | 90.5% (epoch 92) |
| **v5, all 134 words: test top-1 / top-5** | **86.6% / 98.2%** |
| Per test signer top-1 | 79.3, 83.4, 84.3, 87.1, 87.8, 88.0, 88.1, 88.4, 88.6, 88.9, 88.9% (worst P6 79.3%) |
| Words new to the model (96) / words also in Kaggle (38), top-1 over all 134 | 84.0% / 94.0% |

What this shows:
- **The proposal's Goal 1 targets (≥ 75% top-1, ≥ 90% top-5 on unseen signers) are met on the ASL Citizen test signers**, for a 134-word service-counter vocabulary. Still to show: the same on our own camera and users (the other half of the target).
- **The gap between signers is much smaller than on Kaggle**: 79–89% across 11 signers, against 46–79% across 3 for v4 on Kaggle. Webcam data from Deaf signers, the same kind of input the device will see.
- **Kaggle pretraining transfers**: v4 without any ASL Citizen training already got 88% on the shared words (much less domain gap than feared), and the words the model knew are learned better than the new ones (94% vs 84%). How much the pretraining is worth overall is measured by training the same model from random weights (`INIT=scratch`), not done yet.
- Training accuracy (80%) stays below validation (90%) because training sees augmented signs with late dropout and AWP; validation plateaued from about epoch 60, so 100 epochs was enough.
- Not comparable with the ASL Citizen paper's numbers (2,731 signs; ours is a 134-word subset).

Next for v5: calibration and threshold (`calibrate.py`), confusions, live test on the laptop, the `INIT=scratch` comparison, and an "other" class for signs outside the vocabulary.

**Notebook 05 as written (19 Oct 2026):** Gloss variants are merged into the spoken word (WANT1 + WANT2 -> want; 155 glosses -> about 140 words). The v4 weights are kept, the output layer is new (rows of the 38 words that are also in Kaggle copied from v4), 5 epochs output layer only, then everything at 1/5 of the v4 learning rate with the same augmentation, late dropout and AWP; official ASL Citizen signer split, best epoch by validation, test reported once over the 11 test signers. Before training it measures v4 unchanged on the ASL Citizen test videos of the shared words, so the gain from fine-tuning is visible; `INIT=scratch` trains the same model from random weights for comparison. Smoke-tested end to end on synthetic data (CPU), including resume, `calibrate.py` (now reads the signer ids from the logits file, `--meta` optional), `confusions.py` and loading in `SignClassifier`.

## Next steps

In this order:

- **v4 (code ready, `notebooks/04_train_islr_v4.py`), from the Goal 1 review of 14 Oct:** fixes lip mirroring (v1–v3 mirrored the lip points without swapping left and right) and weight decay (v3's was about 200x the 1st-place value); body coordinates (origin between the shoulders, 1 = shoulder width, x rescaled per sign so the camera's aspect ratio drops out, replacing the fixed x 2.08 of the live demo); each hand relative to its wrist; hand-found flags; x-stretch augmentation. All changes in one run because of compute; separate short runs later show which change helped. Compare with v3 on validation only; the body-coordinate change mainly shows on our own camera, so it needs the device test set.

0. **Test fluent splitting more:** 5–10 more recordings with other sign combinations, and recordings by another team member; count correct signs. Then try the Moryossef et al. pretrained model on the same recordings for comparison.
1. **Per-signer accuracy** in the training output (Tran et al. 2023: 81% worry the system works worse for some people).
2. **v3 (code ready, `notebooks/03_train_islr_v3.py`):** the Kaggle 1st-place recipe, read from its code: causal depthwise-conv blocks with channel attention, whole-branch dropout, x/y + 1- and 2-frame motion, speed 0.5–1.5x, shear/rotate/scale/shift, time and space masking, late dropout 0.8 and AWP after epoch 15, RAdam with weight decay 0.1, 300 epochs, early stop after 60 epochs without improvement, resumable after a Colab disconnect, per-signer test accuracy. Same data and signer split as v2. **v4:** add each hand's own coordinates (normalised to its wrist) so hand shape counts more (HELLO vs SUN, WATER).
3. **Step 5, calibration and abstention:** temperature scaling on the validation outputs, then a confidence threshold; evaluate with a risk–coverage curve (Traub et al. 2024). Below the threshold the device asks to repeat instead of speaking.
4. **Compare with the Kaggle 1st-place model** (Hugging Face, MIT licence) on the same webcam; also a speed baseline for the Jetson.
5. **Our own data:** team members and NJIT ASL signers record the chosen vocabulary with our camera; use it to fine-tune and as the real test set ("accuracy on our device").
6. **Step 6:** vocabulary with NJIT ASL. **Step 7:** retrain on ASL Citizen. **Step 8:** hand-off to Goal 2.

## How to resume

- Laptop: `cd /d D:\Multilingual-Sign-to-Speech-Communication-Device`, `.venv\Scripts\activate`, `git pull`, then `python src\live_demo.py --model models\islr_model_v2.pt --vocab ...`
- Colab: mount Drive, copy `MyDrive/sign2speech/islr_*` to `/content`, set `DATA=/content` and `OUT=` the Drive folder, then paste `notebooks/02_train_islr.py` (setup cell is at the top of that file).
