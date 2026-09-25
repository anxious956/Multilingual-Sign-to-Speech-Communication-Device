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

The ECE 414 outline asks the literature review to compare against existing products, with at least 3 patents and 1 paper. Summaries below are from Google Patents. Confirm legal status on each page before submission.

| Patent | Holder | Input → output | How we differ |
|---|---|---|---|
| [US 10,489,639 B2](https://patents.google.com/patent/US10489639B2/en), *Automated sign language translation and communication using multiple input and output modalities* (granted 2019; continuations US 10,956,725 and US 12,183,123) | Avodah | Several cameras plus a depth sensor (structured light or time-of-flight), 3D reconstruction, neural network. Outputs text, speech or an avatar. Two-way. Processing is partly local and partly in the cloud. | We use one RGB camera and no depth sensor, run fully offline, and speak several output languages. |
| [US 2022/0327961 A1](https://patents.google.com/patent/US20220327961A1/en), *Realtime AI Sign Language Recognition* (filed 2021; Google Patents lists a grant as US 12,518,653 B2, not yet verified) | Sign-Speak | Any single camera, body and hand keypoints, a 1D-CNN to find sign boundaries, k-nearest-neighbours with dynamic time warping. **Needs WiFi, with cloud processing.** One embodiment shows the **three most likely translations as a menu** for the user to pick. | Closest prior art. It already covers showing top candidates for the user to pick, so our originality claim can't rest on that alone. Our differences: fully offline on an embedded GPU (no video leaves the device), speech in several languages, and speaking only when confidence passes a threshold or the user confirms. |
| [US 2016/0307469 A1](https://patents.google.com/patent/US20160307469A1/en), *System and Method for Automated Sign Language Recognition* (filed 2016) | Bosch | Gloves with accelerometers and gyroscopes, Hidden Markov Models, a language model. Outputs text or audio in one or more languages. | We need no wearable. That matters because 62% of surveyed Deaf users didn't pick wearables, and sign-language gloves have been criticised by the Deaf community. |
| [US 5,887,069](https://patents.google.com/patent/US5887069) (granted 1999, **expired**) | Hitachi | Data gloves or camera, dynamic-programming pattern matching, text or synthesized speech. Adds facial expression to add emotion to the speech. | Historical baseline. It's expired, so its methods are free to use. |

## Edge deployment (not yet reviewed)

These showed up in the search but haven't been read or vetted yet:

- *SignNet-Nano: Efficient Sign Language Recognition for Real-Time Edge Deployment*: under 20K parameters, benchmarked on Jetson Nano, Xavier NX and Raspberry Pi 4.
- *MP-GestLSTM: real time gesture detection using MediaPipe and LSTM* ([T&F](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2587853)): reports about 25–30 fps on a CPU.

## To do

- [x] Get the ASSETS 2023 paper through the NJIT library and fill in its authors in `references.bib`.
- [ ] Record the ISLR 1st-place leaderboard score from the Kaggle writeup.
- [ ] Check each patent's legal status on Google Patents or USPTO before citing it.
- [ ] Find the published versions of arXiv entries (venue and pages) for the final report.
