# Literature

Papers behind the design decisions in this project, grouped by pipeline stage.

- `pdf/`: papers stored in the repo. Only papers under a CC BY license are here, since that license allows redistribution.
- `references.bib`: BibTeX for every paper below.
- `fetch_papers.sh`: downloads the other open papers into `local/`. That folder is git-ignored, because those papers' licenses don't allow redistribution.

**In repo** means the PDF is in `pdf/`. **Script** means `fetch_papers.sh` downloads it. **Manual** means it's paywalled or needs a login, so get it through the NJIT library or Kaggle and keep it in `local/`.

## 1. Datasets and isolated sign recognition

| Paper | Access | Why it matters |
|---|---|---|
| Desai et al., *ASL Citizen*, NeurIPS 2023 ([arXiv](https://arxiv.org/abs/2304.05934)) | In repo: `pdf/desai2023_asl-citizen.pdf` | Primary dataset: 83,399 videos, 2,731 signs, 52 signers, with an official split by signer. The best baseline (I3D on video) gets 63.1% top-1. The pose-based ST-GCN is a few points lower. Our ≥75% target on 100–250 signs is realistic by comparison. |
| Li et al., *WLASL*, WACV 2020 ([arXiv](https://arxiv.org/abs/1910.11006)) | Script | Benchmark dataset. |
| Sohn, *Google ISLR 1st place: 1D-CNN + Transformer*, Kaggle 2023 ([writeup](https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with), [code](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution)) | Manual (Kaggle) | Reference architecture for our recognition model. It runs on MediaPipe landmarks, the same input our pipeline produces, and the competition limited model size and speed. The dataset has 250 signs from 21 Deaf signers, chosen for the PopSign app (parents learning to sign with young Deaf children), so that vocabulary may not fit our use case. The leaderboard score is still to be verified. |
| *kaggle-asl-signs-1st-place*, Hugging Face ([model](https://huggingface.co/sign/kaggle-asl-signs-1st-place)) | Model download (MIT license) | A LiteRT (TFLite) export of the winning ISLR model, released under MIT. It gives us a ready-made baseline to benchmark on the Jetson before we train our own model. |
| Shi, *Toward ASL Processing in the Real World*, PhD thesis 2023 ([arXiv](https://arxiv.org/abs/2308.12419)) | Script | Covers real-world ASL data, fingerspelling and translation. Useful background. |

## 1b. Training data, transfer and adapting to new signers

Reviewed on 14 Oct 2026 after the v2/v3 results (`docs/GOAL1_LOG.md`): accuracy differs a lot between signers, and some confusions appear only with our own camera. The question was how others get a model to work on new people and new recording conditions.

| Paper / dataset | Access | What it shows | What it means for us |
|---|---|---|---|
| Desai et al., *ASL Citizen*, NeurIPS 2023 ([arXiv](https://arxiv.org/abs/2304.05934)) | In repo | Features from a model trained on WLASL-2000 reach 9.9% top-1 on ASL Citizen, against 58.0% for features trained on ASL Citizen itself; Kinetics (general action) features are near random (Table 4). With 25% of the training videos, recall@1 is 0.16 instead of 0.63 (Table 5). Their pose model (ST-GCN) uses MediaPipe Holistic keypoints, centred and scaled by the shoulder distance, with shear and rotation augmentation. | A model trained on one sign dataset transfers badly to another, even within ASL. Training or fine-tuning on data close to the deployment setting is necessary, and more data matters a lot. Our v4 body normalisation is the same as theirs. |
| Selvaraj et al., *OpenHands*, ACL 2022 ([arXiv](https://arxiv.org/abs/2110.05877)) | Script | Pose-based ISLR models for 6 sign languages. Self-supervised pretraining on unlabelled pose data improves fine-tuning, especially when labelled data is scarce, and pretraining on Indian SL transfers to other sign languages. | Supports "pretrain on a large set, then fine-tune on the target vocabulary": for us, Kaggle ISLR (or PopSign) → ASL Citizen subset → our own recordings. |
| Starner et al., *PopSign ASL v1.0*, NeurIPS 2023 Datasets and Benchmarks ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract.html), [data card](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/)) | CC BY 4.0, videos | The full PopSign data: the same 250 signs as Kaggle ISLR, 47 Deaf signers (split 31 / 8 / 8), about 200,000 videos (about 660 usable "game" videos and 141 "variant" videos per sign), recorded with phone selfie cameras; 1.1 TB of video. An LSTM baseline reaches 82% (validation) / 84% (test). A version 2.1 with more signs is listed. | More than twice as many signers as Kaggle ISLR (21) for the same vocabulary, plus labelled variants of each sign (e.g. both HELLOs). Videos only: landmarks must be extracted, which is the cost. The Kaggle ISLR data is probably a subset of it (same project and signs; not confirmed). |
| Kezar et al., *The Sem-Lex Benchmark*, ASSETS 2023 ([arXiv](https://arxiv.org/abs/2310.00196)) | Download instructions in the paper | Over 84,000 isolated-sign videos from Deaf signers, aligned with ASL-LEX, SignBank and ASL Citizen. Predicting phonological features (handshape, location, movement) as an extra target improves isolated recognition by 2% overall and by 6% in the few-shot case. | Another large consent-based ASL source. Phonological features as an extra training target fit our confusions, which are mostly handshape differences at the same location (THANKYOU/FOOD, PLEASE/MINE). |
| Von Agris et al., signer adaptation on the SIGNUM corpus (RWTH Aachen) ([paper](https://www.bas.uni-muenchen.de/Bas/SIGNUM/papers/paper-02.pdf)) | Open PDF | Adapting a signer-independent recogniser to a new signer with a few labelled examples (eigenvoices, MLLR, MAP): with 10 utterances, about 30% relative error reduction. Older, appearance-based HMM systems. | Even a few labelled signs from a new person help a lot. A short "calibration" round at first use is a reasonable feature. |
| Yin et al., *Weakly Supervised Metric Learning towards Signer Adaptation*, BMVC 2015 ([paper](https://bmva-archive.org.uk/bmvc/2015/papers/paper035/index.html)) | Open | Adapts a learned distance to a new signer using that signer's unlabelled data. | Adaptation without labels is possible in principle. |

**Gap.** No recent pose-based paper was found that fine-tunes or few-shot adapts an isolated-sign recogniser to a new signer and reports the gain. Measuring this on our device (before and after a short per-user calibration) would be a small original contribution, next to selective prediction (section 4).

**Plan that follows (Goal 1):**
1. Pretrain on Kaggle ISLR (done: v3; v4 training). Optionally add PopSign v1.0 for the chosen signs only, to get 47 signers instead of 21.
2. Fine-tune on ASL Citizen for the service-counter vocabulary (52 signers, webcam recordings, signer-independent splits).
3. Fine-tune lightly and recalibrate on our own camera recordings, with some signers held out for testing.
4. Optional: per-user calibration round, measured as above.

## 2. Segmentation (where a sign starts and ends)

| Paper | Access | Why it matters |
|---|---|---|
| Moryossef et al., *Linguistically Motivated Sign Language Segmentation*, Findings of EMNLP 2023 ([arXiv](https://arxiv.org/abs/2310.13960)) | In repo: `pdf/moryossef2023_sign-segmentation.pdf` | In continuous signing there are "minimal to no pauses" between signs, and segmentation needs per-frame begin/inside/outside (BIO) labels. This supports our choice of one sign at a time with a rest pose or button between signs. |

## 3. Signs → sentence (gloss-to-text)

| Paper | Access | Why it matters |
|---|---|---|
| Fayyazsanavi et al., *Gloss2Text*, 2024 ([arXiv](https://arxiv.org/abs/2407.01394)) | In repo: `pdf/fayyazsanavi2024_gloss2text.pdf` | Fine-tunes a pretrained language model to turn signs into sentences. Reference for the language-model option. |
| Zhang & Duh, *Improving Sign Language Gloss Translation with Low-Resource MT Techniques* ([PDF](https://www.cs.jhu.edu/~xzhan138/papers/SLMT_Book_G2T.pdf)) | Script | Describes the recognise-then-translate design we use. Its experiments use German (PHOENIX-2014T) and Chinese (CSL-Daily) data. There is little paired ASL data, which argues for templates plus a small fallback model. |

## 4. Selective prediction (abstaining when unsure)

| Paper | Access | Why it matters |
|---|---|---|
| Traub et al., *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, 2024 ([arXiv](https://arxiv.org/abs/2407.01032)) | Script | How to evaluate a model that can decline to answer: risk–coverage curves. We found no sign-recognition paper that applies this, so our "abstain instead of guess" design is a possible contribution. |

## 5. Deaf community perspectives

| Paper | Access | Why it matters |
|---|---|---|
| Bragg et al., *Sign Language Recognition, Generation, and Translation: An Interdisciplinary Perspective*, ASSETS 2019 ([arXiv](https://arxiv.org/abs/1908.08597)) | Script | The most-cited overview. It says to target specific real-world use cases and to bring in Deaf culture and linguistics expertise. |
| Tran, Ladner & Bragg, *U.S. Deaf Community Perspectives on Automatic Sign Language Translation*, ASSETS 2023 ([ACM](https://dl.acm.org/doi/10.1145/3597638.3614507), [MSR](https://www.microsoft.com/en-us/research/publication/u-s-deaf-community-perspectives-on-automatic-sign-language-translation/)) | Manual (NJIT library) | Survey of 32 U.S. Deaf and hard-of-hearing ASL users. Findings are summarized in [the section below](#tran-et-al-2023-findings). This is our main evidence for choosing the use case. |
| Atwell et al., *"Nothing about us without us"*, 2025 ([arXiv](https://arxiv.org/abs/2512.08839)) | In repo: `pdf/atwell2025_nothing-about-us.pdf` | Concerns about translation accuracy and cultural erosion, and a strong call for Deaf-led design. |

### Tran et al. 2023 findings

The survey had 32 participants. All were Deaf or hard of hearing, and ASL was the primary language for 81% of them. Participants took an ASL check to confirm they sign. The survey covered U.S. users only, and the sample is small.

**Unmet need.** "Wanted an interpreter but couldn't get one" was most frequent in medical settings, then professional, then education.

**Willingness to use automatic translation:**
- Highest: *going into a business*. Every participant showed at least some willingness here.
- Next: self-service kiosks and informal education.
- Lowest: mental healthcare, education settings and formal personal events.
- Medical and mental healthcare split sharply. Most participants answered either 1 or 5.

**Where developers should focus.** *Going into a business* ranked top (mean 3.94 out of 5), then professional settings and social media. The differences between scenarios were small.

**Performance expectations:**
- Accuracy expectations were highest for medical settings (4.9 out of 5) and lowest for self-service kiosks (3.6).
- Participants rated smooth, natural flow (cadence) as more important for automatic translation than for human interpreters, and the difference was statistically significant. Required accuracy and speed didn't differ between the two.

**Hardware:**

| Form factor | Participants comfortable using it |
|---|---|
| Phone app | 91% |
| Stand-alone system (e.g. a computer on a coffee-shop counter) | 69% |
| Wearable | 38% |
| Implant | 6% |

Our device falls in the stand-alone category.

**Top concerns** (reported by more than 80% of participants):

| Concern | Share |
|---|---|
| Hearing people profiting from ASL | 91% |
| Missing ASL grammatical features | 91% |
| Accessibility of the system itself | 88% |
| Automatic translation weakening legal rights to other accommodations | 88% |
| Ignoring the Deaf community's values and needs | 84% |
| Content quality issues | 84% |
| Poorer performance for some users than others | 81% |
| Not recognizing references to the surroundings when describing spatial information | 81% |
| Limited Deaf involvement in leadership or as contributors | 81% |

**Top benefit.** Access in more places or at the last minute (91%).

**What this means for us:**
- A service counter or business setting has the highest willingness, the top developer priority, lower accuracy expectations and a naturally limited vocabulary. That makes it the best fit for a 100–250-sign system.
- Medical settings have the greatest need but the highest accuracy bar and the most divided attitudes. That's risky for a system targeting about 75% top-1.
- "Missing ASL grammatical features" and "weakening legal rights to accommodations" match our scope note. We should keep stating that the device is not a replacement for an interpreter.
- "Poorer performance for some users" means we should report accuracy separately for each signer, not only the average.
- "Limited Deaf involvement" means we should involve Deaf ASL users in choosing the vocabulary and in testing.

## 6. Pipeline components

| Paper | Access | Component |
|---|---|---|
| Lugaresi et al., *MediaPipe*, 2019 ([arXiv](https://arxiv.org/abs/1906.08172)) | Script | Landmark extraction |
| NLLB Team, *No Language Left Behind*, 2022 ([arXiv](https://arxiv.org/abs/2207.04672)) | Script (about 190 pages, so not stored here) | EN → TR/ES translation |
| Radford et al., *Whisper*, 2022 ([arXiv](https://arxiv.org/abs/2212.04356)) | Script | Speech recognition for the reply channel |

## 7. Patents (course requires at least 3)

The ECE 414 outline asks the literature review to compare against existing products, with at least 3 patents and 1 paper. The detailed review is proposal Section 8 (`docs/ECE_414_Proposal_Draft_with_Section8.docx`). Legal status was checked on Google Patents on 26 Oct 2026.

| Patent | Holder | Status (26 Oct 2026) | Input → output | How we differ |
|---|---|---|---|---|
| [US 12,518,653 B2](https://patents.google.com/patent/US12518653B2/en), *Realtime AI sign language recognition* (granted 6 Jan 2026; published as [US 2022/0327961 A1](https://patents.google.com/patent/US20220327961A1/en)) | Sign-Speak | Active, expected to expire 23 Jan 2043 | Any single camera, body and hand keypoints, a 1D-CNN to find sign boundaries, k-nearest-neighbours with dynamic time warping. **The description requires WiFi and a server.** One embodiment shows the **three most likely sentences as a menu** for the user to confirm. | Closest prior art. Our differences: fully offline on an embedded GPU, a calibrated threshold that decides when to show candidates, confirmation of every sentence, speech in three languages and a reply channel. **Granted claim 1 is broad** (body pose network + hand pose network → flattened per-frame features → feature queue → target-language output, no server needed), so a pipeline like ours may fall within it: a commercial version needs a freedom-to-operate review. |
| [US 10,489,639 B2](https://patents.google.com/patent/US10489639B2/en), *Automated sign language translation and communication using multiple input and output modalities* (granted 2019; continuations US 10,956,725 and US 12,183,123) | Avodah | Active (4-year fee paid 2023) | Several cameras plus a depth sensor (structured light or time-of-flight), 3D composite, neural network. Outputs text, speech or an avatar. Two-way, face to face or remote. Processing local, remote or both. | Same face-to-face layout as ours, but every independent claim needs multiple apertures and a depth sensor; we use one RGB camera. |
| [US 12,131,586 B2](https://patents.google.com/patent/US12131586B2/en), *Methods, systems, and machine-readable media for translating sign language content into word content and vice versa* (granted 29 Oct 2024) | Sorenson IP Holdings | Active, expected to expire 12 Feb 2042 | Video from a phone or videophone; translation engine matches sign segments to stored videos, using earlier words and facial cues ("content indicators"); inside a video relay service, AI servers turn ASL into voice and voice into text, with a human or AI "trainer" updating the databases. | Remote calls through a relay service vs our face-to-face counter; servers vs offline; no confidence threshold or confirmation step described. Shows a major relay provider pursuing automatic translation. |
| [US 10,109,219 B2](https://patents.google.com/patent/US10109219B2/en), *System and method for automated sign language recognition* (granted 2018; published as [US 2016/0307469 A1](https://patents.google.com/patent/US20160307469A1/en)) | Bosch | Active (8-year fee paid Apr 2026) | Mainly sensor gloves (camera possible), HMMs with explicit transition models between signs, a language model. Outputs text or audio. | We need no wearable (only 38% of surveyed Deaf users were comfortable with wearables). Its transition models address the same problem as our fluent segmentation. |
| [US 5,887,069](https://patents.google.com/patent/US5887069) (granted 1999) | Hitachi | Expired | Data gloves or camera, dynamic-programming pattern matching, text or synthesized speech; facial expression adds emotion to the speech. | Historical baseline. It's expired, so its methods are free to use. |

## Edge deployment (not yet reviewed)

These showed up in the search but haven't been read or vetted yet:

- *SignNet-Nano: Efficient Sign Language Recognition for Real-Time Edge Deployment*: under 20K parameters, benchmarked on Jetson Nano, Xavier NX and Raspberry Pi 4.
- *MP-GestLSTM: real time gesture detection using MediaPipe and LSTM* ([T&F](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2587853)): reports about 25–30 fps on a CPU.

## To do

- [x] Get the ASSETS 2023 paper through the NJIT library and fill in its authors in `references.bib`.
- [ ] Record the ISLR 1st-place leaderboard score from the Kaggle writeup.
- [x] Check each patent's legal status on Google Patents or USPTO before citing it (26 Oct 2026, table above).
- [ ] Find the published versions of arXiv entries (venue and pages) for the final report. Done for ASL Citizen (NeurIPS 36, pp. 76893–76907, doi:10.52202/075280-3360) and Tran et al. (ASSETS '23, pp. 1–7).
