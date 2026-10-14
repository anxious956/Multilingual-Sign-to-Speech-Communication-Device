## All 250 signs

Test signs: 12888, classes: 250

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 62.3% | 79.5% |
| Signer average | 60.4% | 77.7% |
| Worst signer | 41.2% | 59.2% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 3499 | 41.2% | 59.2% |
| 28656 | 4563 | 65.4% | 83.6% |
| 55372 | 4826 | 74.8% | 90.5% |

**Calibration** (temperature fitted on validation: T = 0.98; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 1.913 | 6.1% |
| After temperature scaling | 1.916 | 7.2% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.151 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.94 | 19.7% | 5.4% |
| ↳ signer 4718 | | 12.3% | 8.1% |
| ↳ signer 28656 | | 17.9% | 6.0% |
| ↳ signer 55372 | | 26.6% | 4.0% |
| 10% | 0.83 | 50.8% | 11.6% |
| ↳ signer 4718 | | 32.1% | 19.7% |
| ↳ signer 28656 | | 50.8% | 11.3% |
| ↳ signer 55372 | | 64.3% | 9.0% |

**Candidate list** (split conformal, 90% target): test coverage 83.3%, list size mean 7.3, median 3; fixed top-5 covers 79.5%.

Figures: `docs\figures\calib_v2_all_250_signs_reliability.png`, `docs\figures\calib_v2_all_250_signs_risk_coverage.png`

## 30-sign vocabulary

Test signs: 1510, classes: 30

| | Top-1 | Top-5 |
|---|---|---|
| All test signs | 81.5% | 90.5% |
| Signer average | 79.7% | 89.2% |
| Worst signer | 60.2% | 75.3% |

| Test signer | Signs | Top-1 | Top-5 |
|---|---|---|---|
| 4718 | 417 | 60.2% | 75.3% |
| 28656 | 523 | 85.5% | 94.1% |
| 55372 | 570 | 93.3% | 98.2% |

**Calibration** (temperature fitted on validation: T = 0.91; T > 1 means the raw model was overconfident)

| | NLL | ECE |
|---|---|---|
| Raw softmax | 0.750 | 3.0% |
| After temperature scaling | 0.746 | 2.1% |

**Abstention** (threshold chosen on validation, measured on test). AURC = 0.029 (lower is better).

| Target error | Threshold | Test coverage (share of signs answered) | Test error among answered |
|---|---|---|---|
| 5% | 0.45 | 82.1% | 6.4% |
| ↳ signer 4718 | | 61.2% | 12.2% |
| ↳ signer 28656 | | 87.0% | 7.5% |
| ↳ signer 55372 | | 92.8% | 2.6% |
| 10% | 0.09 | 99.9% | 18.4% |
| ↳ signer 4718 | | 99.8% | 39.7% |
| ↳ signer 28656 | | 100.0% | 14.5% |
| ↳ signer 55372 | | 99.8% | 6.5% |

**Candidate list** (split conformal, 90% target): test coverage 81.9%, list size mean 1.0, median 1; fixed top-5 covers 90.5%.

Figures: `docs\figures\calib_v2_30_sign_vocabulary_reliability.png`, `docs\figures\calib_v2_30_sign_vocabulary_risk_coverage.png`

**Signs outside the vocabulary** (11378 test signs of the other 220 classes): AUROC (correct in-vocabulary vs outside) = 0.975; at the threshold keeping 95% of correct in-vocabulary signs, 14.7% of outside signs would still be accepted.
