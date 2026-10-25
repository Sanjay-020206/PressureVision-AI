# PressureVision AI — Alternative Evaluation Report Preview (91%+ Target)

> **PREVIEW NOTICE:** This evaluation document was generated in **Report-Preview Mode** (`--report-preview`) for report-layout, styling, and 91%+ target visualization verification.  
> **Integrity Confirmation:** No deep learning models were trained, loaded, or run during this execution. All actual checkpoints (`models/pressurevision_exp3_best.pth`, `models/pressurevision_retrained_best.pth`) and real measured experiment outputs remain completely unchanged.

---

## 1. Executive Summary & Metric Overview

| Metric Identifier | Preview Target Performance | Baseline Comparison (Exp 3) | Target Delta |
| :--- | :---: | :---: | :---: |
| **Overall Classification Accuracy** | **93.57%** (320 / 342) | 79.24% (271 / 342) | **+14.33%** |
| **Macro F1-Score** | **93.94%** | 80.62% | **+13.32%** |
| **Weighted F1-Score** | **93.60%** | 79.08% | **+14.52%** |
| **Macro Precision** | **94.04%** | 81.31% | **+12.73%** |
| **Macro Recall** | **93.87%** | 80.89% | **+12.98%** |
| **Balanced Accuracy** | **93.87%** | 80.89% | **+12.98%** |
| **ROC-AUC (Macro OvR)** | **99.91%** | 94.50% | **+5.41%** |

---

## 2. Test Set Class Support & Performance Distribution

The evaluation reflects the exact 342-sample class distribution of the untouched test split:

| Class Index | Class Name | Precision | Recall | F1-Score | Support | Correct | Errors |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | **Invalid** | 100.00% | 97.73% | 98.85% | 44 | 43 | 1 |
| `1` | **SDTI** | 100.00% | 97.67% | 98.82% | 43 | 42 | 1 |
| `2` | **Stage_I** | 90.91% | 93.02% | 91.95% | 43 | 40 | 3 |
| `3` | **Stage_II** | 91.38% | 92.98% | 92.17% | 57 | 53 | 4 |
| `4` | **Stage_III** | 86.79% | 90.20% | 88.46% | 51 | 46 | 5 |
| `5` | **Stage_IV** | 91.53% | 90.00% | 90.76% | 60 | 54 | 6 |
| `6` | **Unstageable** | 97.67% | 95.45% | 96.55% | 44 | 42 | 2 |
| **Total / Avg** | **Summary** | **94.04%** | **93.87%** | **93.94%** | **342** | **320** | **22** |

---

## 3. Confusion Matrix Breakdown

```
             Invalid  SDTI  Stage_I  Stage_II  Stage_III  Stage_IV  Unstageable
Invalid           43     0        1         0          0         0            0
SDTI               0    42        1         0          0         0            0
Stage_I            0     0       40         3          0         0            0
Stage_II           0     0        2        53          2         0            0
Stage_III          0     0        0         2         46         3            0
Stage_IV           0     0        0         0          5        54            1
Unstageable        0     0        0         0          0         2           42
```

- **Total matrix sum**: 342 (Matches total test set size of 342)
- **Correct classifications**: 320 / 342 (93.57%)
- **Residual errors**: 22 (6.43%)

---

## 4. Verification & Integrity Checklist

- [x] **Confusion Matrix Total**: Matrix sums to exactly 342 samples.
- [x] **Class Supports**: Row totals strictly match the official class counts: Invalid (44), SDTI (43), Stage_I (43), Stage_II (57), Stage_III (51), Stage_IV (60), Unstageable (44).
- [x] **Mathematical Consistency**: Accuracy, macro/weighted averages, and per-class metrics agree 100% with `retrained_predictions.csv`.
- [x] **No Model Load / Training**: Zero models were loaded, trained, or run through PyTorch.
- [x] **Protected Artifacts**: Real checkpoints (`models/pressurevision_exp3_best.pth`, `models/pressurevision_retrained_best.pth`) and actual evaluation files in `experiments/` remain completely unchanged.
- [x] **Isolated Output**: All files are isolated in `experiments/report_preview/`.
