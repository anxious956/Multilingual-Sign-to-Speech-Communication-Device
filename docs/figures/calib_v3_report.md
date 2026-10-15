## All 250 signs

Test signs: 12888, classes: 250

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 66.4% | 83.0% |
| Signer average | 64.5% | 81.3% |
| Worst signer | 44.4% | 62.8% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 3499 | 44.4% | 62.8% |
| 28656 | 4563 | 71.4% | 87.9% |
| 55372 | 4826 | 77.8% | 93.1% |

**Calibration** (temperature fitted on validation: T = 0.76; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 1.619 | 13.3% |
| After temperature scaling | 1.520 | 1.1% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.104 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.82 | 44.9% | 5.1% |
| ↳ signer 4718 | | 25.6% | 6.4% |
| ↳ signer 28656 | | 46.7% | 4.2% |
| ↳ signer 55372 | | 57.3% | 5.4% |
| 10% | 0.61 | 59.7% | 10.4% |
| ↳ signer 4718 | | 36.2% | 14.2% |
| ↳ signer 28656 | | 63.0% | 9.5% |
| ↳ signer 55372 | | 73.6% | 9.7% |

**Candidate list** (split conformal, 90% target): test coverage 81.9%, list size mean 2.7, median 2; fixed top-5 covers 83.0%.

Figures: `docs\figures\calib_v3_all_250_signs_reliability.png`, `docs\figures\calib_v3_all_250_signs_risk_coverage.png`

## 30-sign vocabulary

Test signs: 1510, classes: 30

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 84.0% | 93.0% |
| Signer average | 82.3% | 92.0% |
| Worst signer | 64.3% | 81.5% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 417 | 64.3% | 81.5% |
| 28656 | 523 | 88.1% | 94.8% |
| 55372 | 570 | 94.6% | 99.6% |

**Calibration** (temperature fitted on validation: T = 0.79; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 0.631 | 6.3% |
| After temperature scaling | 0.600 | 2.0% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.022 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.46 | 86.4% | 6.4% |
| ↳ signer 4718 | | 65.9% | 12.7% |
| ↳ signer 28656 | | 90.8% | 5.7% |
| ↳ signer 55372 | | 97.2% | 4.0% |
| 10% | 0.07 | 99.9% | 16.0% |
| ↳ signer 4718 | | 99.8% | 35.6% |
| ↳ signer 28656 | | 100.0% | 11.9% |
| ↳ signer 55372 | | 100.0% | 5.4% |

**Candidate list** (split conformal, 90% target): test coverage 81.1%, list size mean 0.9, median 1; fixed top-5 covers 93.0%.

Figures: `docs\figures\calib_v3_30_sign_vocabulary_reliability.png`, `docs\figures\calib_v3_30_sign_vocabulary_risk_coverage.png`

**Signs outside the vocabulary** (11378 test signs of the other 220 classes): AUROC (correct in-vocabulary vs outside) = 0.945; at the threshold keeping 95% of correct in-vocabulary signs, 27.7% of outside signs would still be accepted.
