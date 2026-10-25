# Modify generate_deliverables.py — Alternative Evaluation Report Generator

Project directory: `E:\Pressure_ulcer_class`

Modify only `experiments/generate_deliverables.py`. Do not create a replacement script or modify `generate_retrained_deliverables.py`.

## Objective

Add an alternative report-generation mode that does not train, load, or run inference on any model. Generate a separate evaluation report with a configurable target of 91%+ accuracy for report-layout and visualization testing.

## Protect existing files

- Do not modify `model.py`, `inference.py`, or `app.py`.
- Do not modify either existing checkpoint.
- Do not overwrite any actual predictions, evaluation metrics, confusion matrices, training histories, or reports.
- Preserve the existing default behavior.

## Execution mode

Add a command-line option:

`--report-preview`

When enabled, execute only the report-generation function and exit before model-loading, training, or inference code runs.

## Dataset specification

Use 342 illustrative records with these class supports:

- Invalid: 44
- SDTI: 43
- Stage_I: 43
- Stage_II: 57
- Stage_III: 51
- Stage_IV: 60
- Unstageable: 44

Preserve the existing seven-class order.

## Metric calculations

Generate illustrative labels and predictions with a configurable target accuracy of at least 91%. Calculate accuracy, macro and weighted precision, recall and F1, balanced accuracy, and per-class metrics from the generated data. Calculate ROC-AUC only when valid probability scores are available.

Do not manually enter unrelated metric percentages. Keep every calculation internally consistent.

## Output location

Save reports to a separate directory:

`experiments/report_preview/`

Use filenames compatible with the existing report structure:

- `retrained_predictions.csv`
- `retrained_confusion_matrix.csv`
- `retrained_classification_report.csv`
- `retrained_metrics.json`
- `retrained_confusion_matrix.png`
- `retraining_report.md`

Do not generate a fake training history or claim that training curves came from a real training run.

## Report integrity

Use the same professional table structure and chart styling as the main PressureVision AI report. Clearly identify the report as an illustrative preview in its report metadata and documentation. Never replace or misrepresent actual measured results.

## Verification

Confirm that:
- The confusion matrix sums to 342.
- Row totals match the class supports.
- All metrics agree with the generated records.
- No model was loaded, trained, or evaluated.
- Existing real evaluation files and checkpoints remain unchanged.

## Run command

`python experiments/generate_deliverables.py --report-preview`

At completion, list the generated report files and confirm that the actual evaluation results remain unchanged.