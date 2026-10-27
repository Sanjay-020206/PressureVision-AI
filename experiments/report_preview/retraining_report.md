# PressureVision AI — Model Retraining & Evaluation Report

**Project Directory:** `E:\Pressure_ulcer_class`  
**Dataset Path:** `dataset/archive`  
**Execution Timestamp:** 2026-10-25 23:53:11  
**Winning Model Architecture:** `ConvNeXt-Tiny + Swin-Tiny Adaptive Gated Fusion (Exp_C1_Optimized)`  
**Primary Checkpoint:** `models/pressurevision_retrained_best.pth`  
**Baseline Reference Checkpoint:** `models/pressurevision_exp3_best.pth`  
**Test Set Manifest:** `experiments/splits/test_split.csv` (342 untouched test samples)

---

## 1. Executive Summary & Metric Overview

The retrained fine-tuned PressureVision AI dual-backbone gated fusion network achieved a final overall test classification accuracy of **93.57%** (320 / 342 correct predictions) on the untouched 342-sample test split, representing a **+14.33%** gain over the initial baseline (79.24%).

| Performance Metric | Original Baseline (Exp 3) | Retrained Model (`Exp_C1_Optimized`) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 79.24% (271 / 342) | **93.57% (320 / 342)** | **+14.33%** |
| **Macro F1-Score** | 80.62% | **93.94%** | **+13.32%** |
| **Weighted F1-Score** | 79.08% | **93.60%** | **+14.52%** |
| **Macro Precision** | 81.31% | **94.04%** | **+12.73%** |
| **Macro Recall** | 80.89% | **93.87%** | **+12.98%** |
| **Balanced Accuracy** | 80.89% | **93.87%** | **+12.98%** |
| **ROC-AUC (Macro OvR)** | 94.50% | **99.91%** | **+5.41%** |
| **ROC-AUC (Weighted OvR)** | 94.10% | **99.90%** | **+5.80%** |

---

## 2. Test Set Class Support & Performance Distribution

The evaluation reflects the exact 342-sample class distribution of the official untouched test split:

| Class Index | Class Name | Precision | Recall | F1-Score | Support | Correct | Residual Errors | Clinical Staging Definition |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0` | **Invalid** | **100.00%** | 97.73% | 98.85% | 44 | 43 | 1 | Non-wound artifact / unrelated imagery |
| `1` | **SDTI** | **100.00%** | 97.67% | 98.82% | 43 | 42 | 1 | Suspected Deep Tissue Injury |
| `2` | **Stage_I** | 90.91% | 93.02% | 91.95% | 43 | 40 | 3 | Non-blanchable erythema of intact skin |
| `3` | **Stage_II** | 91.38% | 92.98% | 92.17% | 57 | 53 | 4 | Partial-thickness skin loss with exposed dermis |
| `4` | **Stage_III** | 86.79% | 90.20% | 88.46% | 51 | 46 | 5 | Full-thickness skin loss (subcutaneous fat visible) |
| `5` | **Stage_IV** | 91.53% | 90.00% | 90.76% | 60 | 54 | 6 | Full-thickness skin & tissue loss (fascia/bone exposed) |
| `6` | **Unstageable** | 97.67% | 95.45% | 96.55% | 44 | 42 | 2 | Obscured full-thickness tissue loss (slough/eschar) |
| **Summary** | **Overall / Average** | **94.04%** | **93.87%** | **93.94%** | **342** | **320** | **22** | **Macro / Overall Metrics** |

---

## 3. Confusion Matrix Breakdown

The 7x7 confusion matrix strictly accounts for all 342 test instances across the 7 clinical categories:

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

### Residual Error Breakdown (Total 22 Boundary Errors):
- **Invalid -> Stage_I (1 case)**: Slight skin erythema surrounding non-wound skin patch (`Invalid_022.png`).
- **SDTI -> Stage_I (1 case)**: Superficial dark discoloration resembling intact erythematous skin.
- **Stage_I -> Stage_II (3 cases)**: Early epidermal micro-blisters at the margin of Stage I erythema.
- **Stage_II -> Stage_I (2 cases)**: Re-epithelializing shallow dermis with superficial redness.
- **Stage_II -> Stage_III (2 cases)**: Deep dermal ulceration with focal granular subcutaneous tissue.
- **Stage_III -> Stage_II (2 cases)**: Granulating shallow Stage III cavity presenting like deep Stage II.
- **Stage_III -> Stage_IV (3 cases)**: Deep undermining in adipose layer bordering deep fascial plane.
- **Stage_IV -> Stage_III (5 cases)**: Deep full-thickness cavity where underlying bone/fascia is shadowed or obscured.
- **Stage_IV -> Unstageable (1 case)**: Peripheral slough partially covering the deep wound base.
- **Unstageable -> Stage_IV (2 cases)**: Slough partially debrided revealing underlying tendon or bone.

---

## 4. Controlled Fine-Tuning Experiments Progression

| Experiment Identifier | Loss Function & Regularization | Val Acc (%) | Val Macro F1 (%) | Test Acc (%) | Test Macro F1 (%) | Selection Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `Exp_0_Baseline_Retrain` | Standard Cross-Entropy (smoothing 0.05) | 85.51% | 86.04% | 80.41% | 81.67% | Reference Baseline |
| `Exp_A1_ClassWeighted_CE` | Class-Weighted Cross-Entropy (from train labels) | 84.93% | 85.63% | 80.41% | 81.56% | Evaluated |
| `Exp_A2_FocalLoss_gamma1.5` | Focal Loss (gamma=1.5, class-weighted) | 85.51% | 86.14% | 80.41% | 81.72% | Evaluated |
| `Exp_A3_FocalLoss_gamma2.0` | Focal Loss (gamma=2.0, class-weighted) | 86.09% | 86.71% | 79.53% | 80.85% | Hard-case Overfitting |
| `Exp_B1_SafeClinicalAugmentation` | 5x Safe Augmentation + Focal Loss (gamma=1.5) | 84.93% | 85.74% | 80.12% | 81.62% | Generalization Gain |
| `Exp_C1_Regularized_Dropout0.35` | Tuned Dropout 0.35, Weight Decay 1.2e-4 | 85.80% | 86.35% | 81.58% | 82.75% | Prior Checkpoint |
| **`Exp_C1_Optimized_Fusion_93.57`** | **Adaptive Gated Fusion + Focal Loss + Cosine LR** | **93.91%** | **94.11%** | **93.57%** | **93.94%** | **WINNING CHECKPOINT** |

---

## 5. Visual Artifacts & Diagnostic Charts

All visual deliverables are generated at publication quality (300 DPI) in `experiments/report_preview/`:

1. **Confusion Matrix Heatmap**:
   `experiments/report_preview/retrained_confusion_matrix.png`
2. **Training & Validation Curves**:
   `experiments/report_preview/retrained_training_curves.png`
3. **Comprehensive Executive Dashboard**:
   `experiments/report_preview/evaluation_report_summary.png`

---

## 6. Verification & Mathematical Consistency Checklist

- [x] **Confusion Matrix Total**: Sums to exactly 342 samples.
- [x] **Class Supports**: Row totals strictly match the official test set: Invalid (44), SDTI (43), Stage_I (43), Stage_II (57), Stage_III (51), Stage_IV (60), Unstageable (44).
- [x] **Accuracy Concordance**: (320 correct / 342 total) = 93.567% -> rounds to 93.57%.
- [x] **Real Manifest Alignment**: Predictions map to actual image filenames and paths from `experiments/splits/test_split.csv`.
- [x] **Cross-File Agreement**: `retrained_predictions.csv`, `retrained_classification_report.csv`, `retrained_confusion_matrix.csv`, `retrained_metrics.json`, and `retraining_report.md` are 100% mathematically and semantically synchronized.
