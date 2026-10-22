## All 250 signs

Test signs: 12888, classes: 250

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 67.2% | 83.9% |
| Signer average | 65.3% | 82.2% |
| Worst signer | 46.2% | 64.7% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 3499 | 46.2% | 64.7% |
| 28656 | 4563 | 71.1% | 88.2% |
| 55372 | 4826 | 78.7% | 93.7% |

**Calibration** (temperature fitted on validation: T = 0.85; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 1.521 | 5.1% |
| After temperature scaling | 1.493 | 2.5% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.101 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.87 | 45.9% | 5.1% |
| ↳ signer 4718 | | 26.7% | 7.4% |
| ↳ signer 28656 | | 47.6% | 4.9% |
| ↳ signer 55372 | | 58.3% | 4.5% |
| 10% | 0.65 | 62.1% | 11.0% |
| ↳ signer 4718 | | 40.5% | 16.4% |
| ↳ signer 28656 | | 64.5% | 10.0% |
| ↳ signer 55372 | | 75.5% | 9.7% |

**Candidate list** (split conformal, 90% target): test coverage 83.0%, list size mean 2.8, median 2; fixed top-5 covers 83.9%.

Figures: `docs\figures\calib_v4_all_250_signs_reliability.png`, `docs\figures\calib_v4_all_250_signs_risk_coverage.png`

## 30-sign vocabulary

Test signs: 1510, classes: 30

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 84.4% | 92.8% |
| Signer average | 82.9% | 91.8% |
| Worst signer | 65.9% | 80.6% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 417 | 65.9% | 80.6% |
| 28656 | 523 | 87.6% | 96.0% |
| 55372 | 570 | 95.1% | 98.9% |

**Calibration** (temperature fitted on validation: T = 0.84; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 0.610 | 4.4% |
| After temperature scaling | 0.600 | 2.4% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.020 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.47 | 87.7% | 7.2% |
| ↳ signer 4718 | | 70.0% | 14.0% |
| ↳ signer 28656 | | 91.2% | 7.5% |
| ↳ signer 55372 | | 97.5% | 3.4% |
| 10% | 0.07 | 100.0% | 15.6% |
| ↳ signer 4718 | | 100.0% | 34.1% |
| ↳ signer 28656 | | 100.0% | 12.4% |
| ↳ signer 55372 | | 100.0% | 4.9% |

**Candidate list** (split conformal, 90% target): test coverage 81.9%, list size mean 0.9, median 1; fixed top-5 covers 92.8%.

Figures: `docs\figures\calib_v4_30_sign_vocabulary_reliability.png`, `docs\figures\calib_v4_30_sign_vocabulary_risk_coverage.png`

**Signs outside the vocabulary** (11378 test signs of the other 220 classes): AUROC (correct in-vocabulary vs outside) = 0.960; at the threshold keeping 95% of correct in-vocabulary signs, 21.7% of outside signs would still be accepted.
