"""
BhashaQuest 2026: 0.80+ Macro F1 Pipeline
Standalone script fine-tuning Google's MuRIL (Multilingual Representations for Indic Languages)
with Stratified 5-Fold Cross Validation & Class-Weighted Loss.
"""

import os, re, glob, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Compute Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

def clean_banglish(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9\s\'\-\.\,\!\?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    train['clean'] = train['text'].apply(clean_banglish)
    test['clean']  = test['text'].apply(clean_banglish)

    classes = sorted(train['label'].unique())
    label2id = {lbl: i for i, lbl in enumerate(classes)}
    id2label = {i: lbl for i, lbl in enumerate(classes)}
    num_labels = len(classes)
    train['label_id'] = train['label'].map(label2id)

    counts = np.bincount(train['label_id'].values)
    class_weights = len(train) / (num_labels * counts.astype(float))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    print(f'Classes: {label2id}')
    print(f'Class counts: {counts}')
    print(f'Class weights for Macro F1: {np.round(class_weights, 3)}')

    # 1. TF-IDF residual
    print('\n[1/3] Computing TF-IDF baseline probabilities...')
    feat = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=60000, sublinear_tf=True, min_df=2)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=80000, sublinear_tf=True, min_df=3)),
    ])
    tfidf_model = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, max_iter=3000, class_weight='balanced'), cv=3
    )
    pipe = Pipeline([('feat', feat), ('clf', tfidf_model)])
    pipe.fit(train['clean'].values, train['label_id'].values)
    tfidf_test_probs = pipe.predict_proba(test['clean'].values)

    # 2. Transformer
    MODEL_NAME = 'google/muril-base-cased'
    MAX_LEN    = 160
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8
    EPOCHS     = 4
    LR         = 2e-5
    N_FOLDS    = 5

    print(f'\n[2/3] Loading {MODEL_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class BanglaDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.encodings = tokenizer(
                list(texts), truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt'
            )
            self.labels = labels

        def __len__(self): return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    test_ds = BanglaDataset(test['clean'].values)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X = train['clean'].values
    y = train['label_id'].values

    oof_probs = np.zeros((len(train), num_labels))
    transformer_test_probs = np.zeros((len(test), num_labels))
    fold_scores = []
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\n--- Fold {fold+1}/{N_FOLDS} ---')
        trn_ds = BanglaDataset(X[trn_idx], y[trn_idx])
        val_ds = BanglaDataset(X[val_idx], y[val_idx])
        trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(trn_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

        best_f1 = 0.0
        best_path = f'best_fold_{fold}.pt'
        best_val_probs = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch in trn_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            model.eval()
            val_preds, val_targets, val_prob_list = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels']
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                    val_preds.extend(probs.argmax(axis=1))
                    val_targets.extend(labels.numpy())
                    val_prob_list.append(probs)

            val_f1 = f1_score(val_targets, val_preds, average='macro')
            print(f'  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(trn_loader):.4f} | Val Macro F1: {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_probs = np.concatenate(val_prob_list, axis=0)
                torch.save(model.state_dict(), best_path)

        fold_scores.append(best_f1)
        oof_probs[val_idx] = best_val_probs
        print(f'  Fold {fold+1} Best Macro F1: {best_f1:.4f}')

        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        test_prob_list = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                test_prob_list.append(probs)

        transformer_test_probs += np.concatenate(test_prob_list, axis=0) / N_FOLDS

        del model, optimizer, scheduler
        if os.path.exists(best_path): os.remove(best_path)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    oof_preds = oof_probs.argmax(axis=1)
    total_oof_f1 = f1_score(y, oof_preds, average='macro')
    print('\n' + '='*50)
    print(f'Overall 5-Fold OOF Macro F1: {total_oof_f1:.4f}')
    print(f'Mean Fold Macro F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}')

    # 3. Ensemble
    print('\n[3/3] Creating Ensemble...')
    final_probs = (0.85 * transformer_test_probs) + (0.15 * tfidf_test_probs)
    pred_ids = final_probs.argmax(axis=1)
    pred_labels = [id2label[i] for i in pred_ids]

    sub = pd.DataFrame({'id': test['id'], 'label': pred_labels})
    sub.to_csv('submission_muril_ensemble.csv', index=False)
    print('[SUCCESS] Saved submission_muril_ensemble.csv (2048 rows)')
    print(sub['label'].value_counts())

if __name__ == '__main__':
    main()

