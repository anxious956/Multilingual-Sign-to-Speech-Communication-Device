## All 134 signs

Test signs: 1913, classes: 134

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 86.7% | 98.2% |
| Signer average | 86.6% | 98.2% |
| Worst signer | 79.3% | 96.6% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 5 | 174 | 79.3% | 96.6% |
| 8 | 189 | 87.8% | 97.9% |
| 13 | 183 | 88.0% | 98.4% |
| 15 | 163 | 83.4% | 97.5% |
| 16 | 173 | 88.4% | 98.8% |
| 19 | 166 | 84.3% | 97.0% |
| 32 | 185 | 88.6% | 98.9% |
| 39 | 171 | 88.9% | 98.8% |
| 44 | 171 | 88.9% | 99.4% |
| 45 | 170 | 87.1% | 99.4% |
| 46 | 168 | 88.1% | 97.6% |

**Calibration** (temperature fitted on validation: T = 0.55; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 0.558 | 14.6% |
| After temperature scaling | 0.424 | 1.5% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.022 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.62 | 85.1% | 6.5% |
| ↳ signer 5 | | 79.9% | 12.9% |
| ↳ signer 8 | | 85.7% | 7.4% |
| ↳ signer 13 | | 86.3% | 5.7% |
| ↳ signer 15 | | 82.2% | 6.7% |
| ↳ signer 16 | | 87.9% | 5.9% |
| ↳ signer 19 | | 81.3% | 6.7% |
| ↳ signer 32 | | 87.6% | 4.3% |
| ↳ signer 39 | | 83.6% | 3.5% |
| ↳ signer 44 | | 86.5% | 4.7% |
| ↳ signer 45 | | 88.8% | 7.9% |
| ↳ signer 46 | | 85.7% | 6.2% |
| 10% | 0.06 | 99.9% | 13.3% |
| ↳ signer 5 | | 100.0% | 20.7% |
| ↳ signer 8 | | 100.0% | 12.2% |
| ↳ signer 13 | | 100.0% | 12.0% |
| ↳ signer 15 | | 100.0% | 16.6% |
| ↳ signer 16 | | 100.0% | 11.6% |
| ↳ signer 19 | | 99.4% | 15.2% |
| ↳ signer 32 | | 100.0% | 11.4% |
| ↳ signer 39 | | 100.0% | 11.1% |
| ↳ signer 44 | | 100.0% | 11.1% |
| ↳ signer 45 | | 100.0% | 12.9% |
| ↳ signer 46 | | 100.0% | 11.9% |

**Candidate list** (split conformal, 90% target): test coverage 87.5%, list size mean 1.0, median 1; fixed top-5 covers 98.2%.

Figures: `docs\figures\calib_v5_all_134_signs_reliability.png`, `docs\figures\calib_v5_all_134_signs_risk_coverage.png`
