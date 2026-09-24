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
| *U.S. Deaf Community Perspectives on Automatic Sign Language Translation*, ASSETS 2023 ([ACM](https://dl.acm.org/doi/10.1145/3597638.3614507), [MSR](https://www.microsoft.com/en-us/research/publication/u-s-deaf-community-perspectives-on-automatic-sign-language-translation/)) | Manual (NJIT library) | U.S. Deaf users on where they'd use sign translation, the accuracy they expect, interface preferences and harms. Needed for choosing the use case. |
| Atwell et al., *"Nothing about us without us"*, 2025 ([arXiv](https://arxiv.org/abs/2512.08839)) | In repo: `pdf/atwell2025_nothing-about-us.pdf` | Concerns about translation accuracy and cultural erosion, and a strong call for Deaf-led design. |

## 6. Pipeline components

| Paper | Access | Component |
|---|---|---|
| Lugaresi et al., *MediaPipe*, 2019 ([arXiv](https://arxiv.org/abs/1906.08172)) | Script | Landmark extraction |
| NLLB Team, *No Language Left Behind*, 2022 ([arXiv](https://arxiv.org/abs/2207.04672)) | Script (about 190 pages, so not stored here) | EN → TR/ES translation |
| Radford et al., *Whisper*, 2022 ([arXiv](https://arxiv.org/abs/2212.04356)) | Script | Speech recognition for the reply channel |

## Edge deployment (not yet reviewed)

These showed up in the search but haven't been read or vetted yet:

- *SignNet-Nano: Efficient Sign Language Recognition for Real-Time Edge Deployment*: under 20K parameters, benchmarked on Jetson Nano, Xavier NX and Raspberry Pi 4.
- *MP-GestLSTM: real time gesture detection using MediaPipe and LSTM* ([T&F](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2587853)): reports about 25–30 fps on a CPU.

## To do

- [ ] Get the ASSETS 2023 paper through the NJIT library and fill in its authors in `references.bib`.
- [ ] Record the ISLR 1st-place leaderboard score from the Kaggle writeup.
- [ ] Find the published versions of arXiv entries (venue and pages) for the final report.
