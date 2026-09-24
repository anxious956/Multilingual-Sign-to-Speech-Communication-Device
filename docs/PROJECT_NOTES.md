# Project Notes

Working notes for the ECE 414 proposal. The course outline is `ECE 414_SLBS_ABET (1).pdf` on `main`.

## Course requirements (ECE 414, Prof. Leonid Tsybeskov)

- **Grading:** proposal presentation 30%, approved proposal report 70%. Missing or late interim reports and insufficient information cost penalty points.
- **Textbook:** Barry Hyman, *Fundamentals of Engineering Design*, 2nd ed.
- **Honor code:** the NJIT Honor Code is enforced. Check the instructor's rules on AI tools before submitting any AI-drafted text. Treat AI drafts as notes and write the final text in the team's own words.

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
- 3–4 figures with captions

Items a–g are a content checklist, not headings. The team chose to write the introduction as continuous prose without subheadings.

## Decisions so far

- **Input language:** ASL only. Turkish Sign Language (TİD) and Spanish Sign Language (LSE) are out of scope.
- **Spoken output:** English, Turkish and Spanish.
- **Reply channel:** the hearing person speaks EN, TR or ES, and the Deaf user reads English text.
- **Pipeline order:** signs → English sentence → user confirms in English → translate → speak. The confirmation happens in English because that's the language the Deaf user can check.
- **Sign segmentation:** one sign at a time, separated by a rest pose, with a button as backup. Moryossef et al. 2023 supports this.
- **Abstract:** already submitted. It names Deaf users, hospitals, schools and public services.

## Open decisions

- **Use case and vocabulary.** The recommendation is a service counter or business setting (Tran et al. 2023: highest willingness, top developer priority, lower accuracy expectations). Hospitals stay as motivation, framed as front desk or check-in use, or use while waiting for an interpreter.
- **Signs → sentence:** rules and templates, a small language model, or both.
- **Latency definition:** where the 1.2 s clock starts and stops. It can't include the user's confirmation time.
- **Fingerspelling:** core feature or stretch goal.

## Originality (from the patent review)

Sign-Speak's patent (US 2022/0327961 A1) already shows users the top 3 candidate translations as a menu, so a candidate list alone isn't new. Our differences:

1. Runs fully offline on an embedded GPU, and no video leaves the device.
2. Needs one ordinary RGB camera: no gloves or depth sensors.
3. Speaks several output languages.
4. Stays silent when unsure: it speaks only when a calibrated confidence threshold is passed or the user confirms.

## To do

- [ ] Get the team's approval of the introduction draft, ideally as a tracked-changes version on the teammate's original text.
- [ ] Write the remaining introduction parts: c, d, e and f.
- [ ] Make 3–4 figures, e.g. a block diagram, a pipeline flow chart, the confirmation UI and a comparison table.
- [ ] Marketability and comparison with existing products (SignAll, Sign-Speak, Avodah).
- [ ] Constraints: technical, legal (ADA and interpreter rights, privacy) and budget (Jetson and parts cost list).
- [ ] Specification table, building on the targets table in `README.md`.
- [ ] Environmental and economic factors: power use, e-waste, device cost versus interpreter cost.
- [ ] Verify each patent's legal status on Google Patents or USPTO.
- [ ] Get the ISLR 1st-place leaderboard score from Kaggle (needs a team member's login).
- [ ] Move the reference list to section 12 in the final report.

## Related files

- `literature/README.md`: papers and patents, with summaries.
- `literature/references.bib`: BibTeX for all sources.
- `docs/Introduction_Draft.docx`: draft of intro parts a and b.
