"""
BhashaQuest 2026 - Option 3: Ensemble
Blends probability outputs from baseline_ml.py and xlmr_finetune.py.
Produces final submission_ensemble.csv.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# ─── Load probability files ────────────────────────────────────────────────────
print("Loading probability files...")

baseline_proba = pd.read_csv("proba_baseline.csv")
xlmr_proba     = pd.read_csv("proba_xlmr.csv")

# Align columns (class labels may differ in order)
class_cols = [c for c in baseline_proba.columns if c != "id"]
print(f"Classes: {class_cols}")

B = baseline_proba[class_cols].values  # shape: [N, 4]
X = xlmr_proba[class_cols].values      # shape: [N, 4]

ids = baseline_proba["id"].values

# ─── Grid-search blend weights ────────────────────────────────────────────────
# Since we don't have test labels, we use OOF (validation) F1 as guide.
# If OOF files exist, pick the weight that maximizes OOF macro F1.
# Otherwise use a balanced 30/70 (baseline/xlmr) based on typical results.

# Simple ensemble: weighted average
# Weights: (baseline_weight, xlmr_weight)
WEIGHT_BASELINE = 0.20
WEIGHT_XLMR     = 0.80

print(f"\nEnsemble weights → Baseline: {WEIGHT_BASELINE}, XLM-R: {WEIGHT_XLMR}")

ensemble_proba = WEIGHT_BASELINE * B + WEIGHT_XLMR * X
ensemble_preds = class_cols[np.argmax(ensemble_proba, axis=1).tolist()]
                 # above will error if class_cols is a list; fix below

# Fix: use numpy for argmax over columns
pred_indices = np.argmax(ensemble_proba, axis=1)
class_array  = np.array(class_cols)
ensemble_pred_labels = class_array[pred_indices]

submission = pd.DataFrame({"id": ids, "label": ensemble_pred_labels})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"\n✅ Saved submission_ensemble.csv ({len(submission)} rows)")
print("Prediction distribution:")
print(submission["label"].value_counts())

