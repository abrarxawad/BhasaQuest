"""
BhashaQuest 2026 - Option 1: TF-IDF + Classical ML Baseline
Produces submission_baseline.csv for immediate Kaggle upload.
"""

import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def preprocess(text):
    """Minimal but effective preprocessing for Romanized Bangla."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize repeated characters (e.g., "korrte" -> "korte")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s\'\-\.\,\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # ─── 1. Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print("Label distribution:\n", train["label"].value_counts())

    # ─── 2. Preprocessing ─────────────────────────────────────────────────────────
    train["clean"] = train["text"].apply(preprocess)
    test["clean"]  = test["text"].apply(preprocess)

    X_train = train["clean"].values
    y_train = train["label"].values
    X_test  = test["clean"].values

    # ─── 3. Feature Engineering: Word + Char TF-IDF ───────────────────────────────
    print("\nBuilding TF-IDF features...")
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2,
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
    )

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    # ─── 4. Train & Evaluate ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Model 1: Logistic Regression ---")
    lr_pipe = Pipeline([
        ("feat", features),
        ("clf",  LogisticRegression(C=5.0, max_iter=2000, solver="lbfgs",
                                     class_weight="balanced")),
    ])
    lr_scores = cross_val_score(lr_pipe, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=2)
    print(f"  5-Fold Macro F1: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")

    print("\n--- Model 2: LinearSVC (Calibrated) ---")
    svc_pipe = Pipeline([
        ("feat", features),
        ("clf",  CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, max_iter=3000, class_weight="balanced"),
            cv=3
        )),
    ])
    svc_scores = cross_val_score(svc_pipe, X_train, y_train, cv=cv,
                                  scoring="f1_macro", n_jobs=2)
    print(f"  5-Fold Macro F1: {svc_scores.mean():.4f} +/- {svc_scores.std():.4f}")

    # ─── 5. Train best model on full training set ─────────────────────────────────
    if lr_scores.mean() >= svc_scores.mean():
        best_name = "Logistic Regression"
        best_pipe  = lr_pipe
    else:
        best_name = "LinearSVC"
        best_pipe  = svc_pipe

    print(f"\nBest model: {best_name}")
    print("Training on full dataset...")
    best_pipe.fit(X_train, y_train)

    # ─── 6. Predict & Save ────────────────────────────────────────────────────────
    print("Generating predictions...")
    preds = best_pipe.predict(X_test)

    submission = pd.DataFrame({"id": test["id"], "label": preds})
    submission.to_csv("submission_baseline.csv", index=False)
    print(f"\n[SUCCESS] Saved submission_baseline.csv ({len(submission)} rows)")
    print("Label distribution in predictions:")
    print(submission["label"].value_counts())

    # Also save probabilities for later ensembling
    proba = best_pipe.predict_proba(X_test)
    classes = best_pipe.classes_
    proba_df = pd.DataFrame(proba, columns=classes)
    proba_df.insert(0, "id", test["id"].values)
    proba_df.to_csv("proba_baseline.csv", index=False)
    print("[SUCCESS] Saved proba_baseline.csv (for ensembling)")

if __name__ == "__main__":
    main()
