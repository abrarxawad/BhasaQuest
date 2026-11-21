"""
BhashaQuest 2026 - Option 2 & 3: XLM-RoBERTa Fine-tuning
Requires: pip install transformers torch datasets
Produces submission_xlmr.csv and proba_xlmr.csv for ensembling.
"""

import re, os, random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "xlm-roberta-base"   # Will be downloaded if not cached
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
N_FOLDS      = 5
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Seed ─────────────────────────────────────────────────────────────────────
def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all()

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text.strip()

train["clean"] = train["text"].apply(preprocess)
test["clean"]  = test["text"].apply(preprocess)

# Encode labels
label2id = {lbl: i for i, lbl in enumerate(sorted(train["label"].unique()))}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
print(f"Classes: {label2id}")

train["label_id"] = train["label"].map(label2id)

# ─── Dataset ──────────────────────────────────────────────────────────────────
class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
    def __len__(self): return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

# ─── Training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return f1_score(all_labels, all_preds, average="macro")

def predict_proba(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

# ─── K-Fold Training ──────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_dataset = TestDataset(test["clean"].values, tokenizer)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X = train["clean"].values
y = train["label_id"].values

oof_probs   = np.zeros((len(train), num_labels))
test_probs  = np.zeros((len(test),  num_labels))
fold_scores = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nFold {fold+1}/{N_FOLDS}")

    trn_ds = BanglaDataset(X[trn_idx], y[trn_idx], tokenizer)
    val_ds = BanglaDataset(X[val_idx], y[val_idx], tokenizer)

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
        label2id=label2id, id2label=id2label
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps   = len(trn_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1   = 0.0
    best_path = f"best_fold{fold+1}.pt"

    for epoch in range(EPOCHS):
        loss = train_epoch(model, trn_loader, optimizer, scheduler)
        f1   = eval_epoch(model, val_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val Macro F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)

    print(f"  Best Val Macro F1 for fold {fold+1}: {best_f1:.4f}")
    fold_scores.append(best_f1)

    # Load best checkpoint and get OOF + test predictions
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    val_ds_no_shuf = BanglaDataset(X[val_idx], y[val_idx], tokenizer)
    val_loader_no_shuf = DataLoader(val_ds_no_shuf, batch_size=BATCH_SIZE * 2,
                                    shuffle=False)
    # OOF
    model.eval()
    oof_fold = []
    with torch.no_grad():
        for batch in val_loader_no_shuf:
            batch.pop("labels")
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            oof_fold.append(probs)
    oof_probs[val_idx] = np.concatenate(oof_fold, axis=0)

    # Test
    test_probs += predict_proba(model, test_loader) / N_FOLDS

    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    os.remove(best_path)

# ─── Results ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Mean CV Macro F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
oof_preds = oof_probs.argmax(axis=1)
oof_f1    = f1_score(y, oof_preds, average="macro")
print(f"OOF Macro F1 (combined): {oof_f1:.4f}")

# ─── Save predictions ─────────────────────────────────────────────────────────
test_pred_labels = [id2label[i] for i in test_probs.argmax(axis=1)]
submission = pd.DataFrame({"id": test["id"], "label": test_pred_labels})
submission.to_csv("submission_xlmr.csv", index=False)
print(f"\n✅ Saved submission_xlmr.csv ({len(submission)} rows)")

# Save probabilities (sorted by label2id order for ensembling)
sorted_classes = [id2label[i] for i in range(num_labels)]
proba_df = pd.DataFrame(test_probs, columns=sorted_classes)
proba_df.insert(0, "id", test["id"].values)
proba_df.to_csv("proba_xlmr.csv", index=False)
print("✅ Saved proba_xlmr.csv (for ensembling)")

