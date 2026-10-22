# Project Notes

Working notes for the ECE 414 proposal. The course outline and lecture notes are in `lecture-notes/`.

## Course requirements (ECE 414, Prof. Leonid Tsybeskov)

- **Grading:** proposal presentation 30%, approved proposal report 70%. Missing or late interim reports and insufficient information cost penalty points.
- **Textbook:** Barry Hyman, *Fundamentals of Engineering Design*, 2nd ed.
- **Honor code:** the NJIT Honor Code is enforced. The team confirmed that AI use is allowed for the final project.

### Schedule

| Weeks | Deliverable | Interim report |
|---|---|---|
| 1–3 | Preliminary description and title, team formation, division of responsibilities | Yes |
| 4 | Final title and technical goals (needs instructor approval) | |
| 5 | Constraints: technical, legal, budgetary | Yes |
| 6 | Design methodology | Yes |
| 7–8 | Literature review: originality, marketability, comparison with existing products (**at least 3 patents and 1 paper**) | Yes |
| 9 | Technical documentation: flow chart, block diagram, specification table | Yes |
| 10–12 | Proposal defense | |
| 12–14 | Proposal report | |

### ABET outcomes to cover

The report must show consideration of public health, safety and welfare, and of global, cultural, social, environmental and economic factors (outcome 2). It must also show ethical and professional responsibility (outcome 4).

### Introduction requirements (report section 3)

The introduction starts on page 3 and must include:

- a) State of the art
- b) The problem addressed
- c) Objectives
- d) Adopted approach
- e) Current trends
- f) Commercial systems, including research status, ethics, economics and sustainability
- g) Formal citations to the reference list in section 12
- **At least 5 pages of text, 3–5 figures, 1–2 tables** and references (announcement, `lecture-notes/introduction/01_announcement_intro_requirements.jpeg`)

Items a–g are a content checklist, not headings. Following Zeynep's review, the introduction uses subheadings 3.1 Background, 3.2 Problem Statement, 3.3 Objectives, 3.4 Proposed Approach, 3.5 Current Trends, 3.6 Existing Systems and 3.7 Ethics, Economics and Sustainability, so the instructor can find each item.

Template rules (from `ECE_414_Proposal_Template_2026_2.docx`): references in the format "Authors. Title. Journal, volume(issue), pages, year.", numbered in order of appearance, with the number at the end of the sentence; figure captions below as "Fig. N."; table titles above tables; every number has units; every acronym spelled out at first use.

## Decisions so far

- **Input language:** ASL only. Turkish Sign Language (TİD) and Spanish Sign Language (LSE) are out of scope.
- **Spoken output:** English, Turkish and Spanish.
- **Reply channel:** the hearing person speaks EN, TR or ES, and the Deaf user reads English text.
- **Pipeline order:** signs → English sentence → user confirms in English → translate → speak. The confirmation happens in English because that's the language the Deaf user can check.
- **Sign segmentation:** one sign at a time, separated by a rest pose, with a button as backup. Moryossef et al. 2023 supports this. Since October 2026 the prototype can also split a few signs made without lowering the hands (hand slow-downs as candidate cuts, chosen by the sign model; see `docs/GOAL1_LOG.md`), still being tested.
- **Abstract:** already submitted. It names Deaf users, hospitals, schools and public services.
- **Use case:** a service counter or front desk (Tran et al. 2023: highest willingness, lower accuracy expectations). Hospitals stay as motivation, framed as front-desk use or use while waiting for an interpreter.
- **Signs → sentence:** templates for common phrases, with a small language model as a fallback.
- **Latency:** under 1.2 s, measured from the user's confirmation to the start of speech. The confirmation time itself is not counted.
- **Fingerspelling:** a fallback for words outside the vocabulary, such as names, treated as a stretch goal (Section 4.1, Table 3); it is not part of the accuracy targets.
- **Connectivity:** fully offline. No video leaves the device, which avoids privacy risk and works without a network; this is also the main difference from Sign-Speak's cloud-based patent.

## Open decisions

- **Vocabulary list:** which 100–250 signs to cover for the service-counter setting, ideally chosen with Deaf ASL users.

## Originality (from the patent review)

Sign-Speak's patent (US 2022/0327961 A1) already shows users the top 3 candidate translations as a menu, so a candidate list alone isn't new. Our differences:

1. Runs fully offline on an embedded GPU, and no video leaves the device.
2. Needs one ordinary RGB camera: no gloves or depth sensors.
3. Speaks several output languages.
4. Stays silent when unsure: it speaks only when a calibrated confidence threshold is passed or the user confirms.

## To do

- [ ] Get the team's approval of the introduction draft, ideally as a tracked-changes version on the teammate's original text.
- [x] Write the remaining introduction parts: c, d, e and f (draft).
- [x] Replace the figures with sourced ones, following the sample: a survey chart redrawn from Tran et al. data, our service-counter drawing, the Sign-Speak patent drawing and the Atwell et al. concerns chart (CC BY). Tables use a classic three-line style.
- [x] Expand the introduction to at least 5 pages of text (about 1,930 words at 1.5 line spacing) with 4 figures and 2 tables.
- [x] Section 4: three technical goals drafted (sign recognition, language and speech, offline embedded integration).
- [x] Section 4: Table 4 names filled in (Goal 1: Asmar Hasanova; Goal 2: Dorukhan Cakir and Zeynep Hafsa Cakici; Goal 3: Merrick Simmons; project-wide tasks shared by the whole team).
- [ ] Cover page: fill in the team number.
- [x] Section 5: community impact and ethical issues drafted, citing the ADA and the IEEE Code of Ethics.
- [ ] Contact the ASL community at NJIT before choosing the vocabulary (promised in Section 5).
- [ ] Buy the NVIDIA Jetson Orin Nano.
- [ ] Decide on the power supply (battery or wall power) after measuring power draw.
- [ ] Marketability and comparison with existing products (SignAll, Sign-Speak, Avodah).
- [ ] Constraints: technical, legal (ADA and interpreter rights, privacy) and budget (Jetson and parts cost list).
- [ ] Specification table, building on the targets table in `README.md`.
- [ ] Environmental and economic factors: power use, e-waste, device cost versus interpreter cost.
- [ ] Verify each patent's legal status on Google Patents or USPTO.
- [ ] Get the ISLR 1st-place leaderboard score from Kaggle (needs a team member's login).
- [x] Move the reference list to Section 12 (done in the combined proposal file).
- [x] Section 7 (draft): design flow chart on the course template (Fig. 5), hardware block diagram (Fig. 6), prototype categories P1–P5 (Table 14), validation and qualification categories (Table 15), verification matrix (Table 16), preliminary data and simulations (Figs. 7–9, Table 13). Needs the team's review, especially the block diagram (Merrick) and the Goal 2 timings (Dorukhan, Zeynep).
- [ ] Have native speakers check the NLLB-200 test translations in `docs/figures/section7_benchmarks.json` (one Turkish word was misspelled; three outputs began with a stray dash).
- [ ] Test the Turkish reply channel with recorded voices: Whisper small had 12.5% word error rate on synthetic Turkish speech, against 0% English and 2.4% Spanish.

## Related files

- `literature/README.md`: papers and patents, with summaries.
- `literature/references.bib`: BibTeX for all sources.
- `docs/ECE_414_Proposal_Draft.docx`: combined proposal (cover page, abstract, table of contents, Sections 3, 4, 5 and 12). This is the single source: it is built on the team's own latest Word file, with Sections 4 and 5 added. Includes student ID numbers (the team chose to publish them).
- `docs/ECE_414_Proposal_Draft_with_Section7.docx` (+ `.pdf`): the Section 6 file with Section 7 added, built by `tools/add_section7_to_proposal.py` (text and tables in `tools/section7_content.py`). Edit the content file and rerun rather than editing the Word file, or make later edits in Word once Section 7 is final.
- `docs/figures/`: figure images and the scripts that draw Figure 1 (`make_survey_chart.py`), Figure 2 (`make_setup_sketch.py`) and Figures 5–10 (`make_section7_figures.py`). `section7_benchmarks.json` holds the Section 7.5 measurements, written by `tools/bench_section7.py` (CPU only; recognizer size and speed, NLLB-200, Piper and Whisper timings, correction simulation, memory budget).
- `docs/GOAL1_LOG.md`: Goal 1 (sign recognition) progress, model results and next steps.
- `src/README.md`: how to set up and run the Goal 1 scripts on a laptop.
