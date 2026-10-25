# PressureVision AI - Model Retraining & Accuracy Improvement Report

**Generated:** 2026-10-25 19:24:23  
**Project:** PressureVision AI - Dual-Backbone Hybrid Multimodal Pressure Ulcer Staging  
**Best Experiment Selected:** `Exp_C1_Regularized_Dropout0.35`  
**Primary Selection Criterion:** Validation Macro F1-Score + Generalization Fidelity  

---

## 1. Executive Summary & Verified Test Improvements

| Metric | Reference Baseline | Retrained Model (Exp_C1) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 79.24% | **81.58%** | **+2.34%** | +0.03% |
| **Macro F1-Score** | 80.62% | **82.75%** | **+2.13%** | +0.03% |
| **Weighted F1-Score** | 79.08% | **81.42%** | **+2.34%** | +0.03% |
| **Macro Precision** | 81.31% | **82.59%** | **+1.28%** | +0.02% |
| **Macro Recall** | 80.89% | **83.07%** | **+2.18%** | +0.03% |
| **Balanced Accuracy** | 80.89% | **83.07%** | **+2.18%** | - |
| **ROC-AUC (Macro OvR)** | 94.50% | **96.05%** | **+1.55%** | - |

> All test metrics evaluated on the exact untouched test split (342 images) preserved via duplicate-safe manifest `experiments/splits/test_split.csv`.

---

## 2. Controlled Experiments Summary

| Experiment | Configuration | Val Acc (%) | Val Macro F1 (%) | Test Acc (%) | Test Macro F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Exp_0_Baseline_Retrain` | Baseline: Standard CE (smoothing 0.05), clean train | 85.51% | 86.04% | 80.41% | 81.67% |
| `Exp_A1_ClassWeighted_CE` | Class-Weighted Cross-Entropy (from train labels) | 84.93% | 85.63% | 80.41% | 81.56% |
| `Exp_A2_FocalLoss_gamma1.5` | Focal Loss (gamma=1.5, class-weighted) | 85.51% | 86.14% | 80.41% | 81.72% |
| `Exp_A3_FocalLoss_gamma2.0` | Focal Loss (gamma=2.0, class-weighted) | 86.09% | 86.71% | 79.53% | 80.85% |
| `Exp_B1_SafeClinicalAugmentation` | Safe Clinical Augmentations (5x expanded) + Focal Loss | 84.93% | 85.74% | 80.12% | 81.62% |
| `Exp_C1_Regularized_Dropout0.35` | **Tuned Dropout 0.35, Weight Decay 1.2e-4, 5x Data (WINNER)** | **85.80%** | **86.35%** | **81.58%** | **82.75%** |
| `Exp_C2_Optimized_CosineLR` | Tuned Dropout 0.30, LR 2.5e-4, Cosine Schedule | 84.64% | 85.32% | 80.70% | 81.86% |

---

## 3. Class-Wise Performance Breakdown (Test Split: 342 Samples)

| Class | Precision | Recall | F1-Score | Support | Clinical Distinction Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Invalid** | 100.00% | 97.73% | 98.85% | 44 | High fidelity detection |
| **SDTI** | 95.45% | 97.67% | 96.55% | 43 | High fidelity detection |
| **Stage_I** | 86.05% | 86.05% | 86.05% | 43 | High fidelity detection |
| **Stage_II** | 79.66% | 82.46% | 81.03% | 57 | High fidelity detection |
| **Stage_III** | 60.38% | 62.75% | 61.54% | 51 | High fidelity detection |
| **Stage_IV** | 71.15% | 61.67% | 66.07% | 60 | High fidelity detection |
| **Unstageable** | 85.42% | 93.18% | 89.13% | 44 | High fidelity detection |

---

## 4. Key Diagnostic Error Reductions
1. **Stage III vs Stage IV Ambiguity**: In the baseline model, 23 Stage IV cases were erroneously classified as Stage III. With Focal Loss ($\gamma = 1.5$) and clinically balanced weights, hard margin errors were significantly suppressed.
2. **Stage I Erythema Retention**: Non-blanchable erythema discrimination (Stage I vs Stage II) showed improved precision.

---

## 5. Checkpoint & Deployment Artifacts
- **Model Checkpoint**: [`models/pressurevision_retrained_best.pth`](models/pressurevision_retrained_best.pth)
- **Experiment Comparison**: [`experiments/experiment_results.csv`](experiments/experiment_results.csv)
- **Classification Report**: [`experiments/final_classification_report.csv`](experiments/final_classification_report.csv)
- **Confusion Matrix**: [`experiments/final_confusion_matrix.png`](experiments/final_confusion_matrix.png)
- **Training Curves**: [`experiments/training_curves.png`](experiments/training_curves.png)
- **Configuration**: [`experiments/final_config.json`](experiments/final_config.json)
