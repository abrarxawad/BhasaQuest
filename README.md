# Intra 2K23 NLP Competition: BhashaQuest 2026
## Classification of Informal Romanized Bangla (Banglish) Text
**Organized by: K-MINDS (Batch 2K23)**  
**Author:** Abrar Jawad  
**Task:** 4-Class Text Classification on Romanized Bangla  
**Evaluation Metric:** Macro F1 Score  

---

## 1. Problem Statement & Motivation

Digital communication in Bangladesh is dominated by **Romanized Bangla** (popularly known as *Banglish*)—the practice of writing the Bengali language using the Latin alphabet (English characters) rather than standard Bengali Unicode script (e.g., *"Ami bhalo achi"* instead of *"আমি ভালো আছি"*). 

Despite its ubiquitous prevalence in social media, messaging apps, customer service portals, and online forums, Romanized Bangla poses severe Natural Language Processing (NLP) challenges:
1. **Lack of Standardized Orthography:** There is no official dictionary or standardized phonetic spelling rules for Banglish. Writers spell phonetically according to personal typing habits, dialect, or keyboard shortcuts (e.g., *"korte"*, *"kortei"*, *"kortey"*, *"korta"*).
2. **Phonetic & Morphological Ambiguity:** Multiple distinct Bengali phonemes map onto single English letters (e.g., 's', 'sh', and 'ss' for শ, ষ, স; 't' for ত and ট; 'd' for দ and ড).
3. **Informal & Code-Mixed Structures:** Conversations frequently blend colloquial abbreviations, internet slang, and English technical terms.
4. **Context-Dependent Intent:** Words taken out of context fail to convey safety, maliciousness, or query intent.

### The Competition Challenge
The objective of the **Intra 2K23 NLP Competition (BhashaQuest 2026)** is to automatically classify informal Romanized Bangla text samples into one of **four anonymous predefined categories** (`14AC78`, `72A2C8`, `19E1A1`, `6C0197`). Submissions are evaluated strictly using the **Macro F1 Score**, requiring balanced precision and recall across all classes regardless of class frequency.

---

## 2. Dataset Analysis & Class Decryption

### 2.1 Dataset Composition
* **Training Set (`train.csv`):** 8,195 samples with columns `id`, `text`, and `label`.
* **Test Set (`test.csv`):** 2,048 unlabelled samples with columns `id` and `text`.
* **Submission Format (`sample_submission.csv`):** Two columns: `id`, `label`.

### 2.2 Class Distribution Analysis
Analysis of `train.csv` revealed the following class frequencies:

| Class Identifier | Sample Count | Percentage | Class Characteristics & Inferred Domain |
| :--- | :---: | :---: | :--- |
| **`14AC78`** | 2,791 | 34.06% | **Benign / Customer Service / E-Commerce Queries:** Inquiries regarding product orders, returns, general knowledge, cooking recipes, and technical questions (*"order"*, *"ponyoti"*, *"ferot"*, *"bikretar kache"*). |
| **`72A2C8`** | 2,155 | 26.30% | **Prompt Injection / Jailbreak / LLM Adversarial Attacks:** LLM safety testing phrases trying to override system prompts (*"upeksha korun"*, *"uporer nirdesh"*, *"prompt leak"*, *"rot13"*, *"system instructions"*). |
| **`19E1A1`** | 1,994 | 24.33% | **Severe Malicious / Cybercrime / High-Risk Harm:** Explicit harmful requests involving cyber-attacks, physical violence, weapons, chemical harm, and forensic case briefs (*"cyanide"*, *"acid"*, *"cyber akromon"*, *"bmdc"*, *"forensic"*). |
| **`6C0197`** | 1,255 | 15.31% | **Abusive Speech / Toxicity / Demographic Slurs:** Targeted harassment, profanity, sexist, religious, or racial discrimination (*"boka"*, *"kutta"*, *"sada/kalo"*, *"somokami"*, *"purushra"*). |

```
Class Distribution in Training Set:
14AC78  [==============================]  2,791 (34.1%)
72A2C8  [=======================]        2,155 (26.3%)
19E1A1  [=====================]          1,994 (24.3%)
6C0197  [=============]                  1,255 (15.3%)
```

**Key Takeaway for Optimization:**  
Because evaluation uses **Macro F1**, the smallest class (`6C0197` with only 1,255 samples) contributes exactly 25% of the final leaderboard score. Standard models optimizing overall accuracy tend to under-predict the minority class, severely hurting Macro F1. Consequently, class-weighted training and decision threshold calibration are essential.

---

## 3. Methodology & Architecture

Our solution pipeline was systematically developed through an iterative engineering approach:

```
[Raw Romanized Bangla Text]
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Domain-Specific Text Cleaning & Normalization        │
│  - Lowercasing                                         │
│  - Character-repetition normalization (e.g. 'koorrrte') │
│  - Preserving syntactic punctuation (?, !)             │
└────────────────────────────────────────────────────────┘
            │
            ├────────────────────────────────────────────┐
            ▼                                            ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Feature Extraction Pipeline           │   │ Deep Pretrained Transformer Pipeline  │
│ - Word n-grams: (1, 3)                │   │ - XLM-RoBERTa-base                    │
│   max_features=80,000, sublinear_tf   │   │ - Sequence length: 128 / 140          │
│ - Char n-grams: (2, 5) char_wb        │   │ - Dynamic Class-Weighted CrossEntropy │
│   max_features=100,000, sublinear_tf  │   │ - FP16 Mixed Precision                │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
            │                                            │
            ▼                                            ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Multi-Class Statistical Classifiers   │   │ Stratified 5-Fold Cross Validation    │
│ - Calibrated LinearSVC (C=1.0)        │   │ - Out-of-fold probability estimation  │
│ - Balanced Logistic Regression        │   │ - ModelCheckpoint selection           │
│ - Complement Naive Bayes              │   └───────────────────────────────────────┘
└───────────────────────────────────────┘                │
            │                                            │
            ▼                                            ▼
┌────────────────────────────────────────────────────────┐
│  Post-Processing & Optimization Layer                  │
│  - Out-of-Fold Soft Voting / Stacking Ensemble         │
│  - Prior-Compensated Decision Threshold Optimization   │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Final Submission Prediction Output (`submission.csv`) │
└────────────────────────────────────────────────────────┘
```

---

## 4. Data Preprocessing & Cleaning Pipeline

Due to the informal nature of online chat and social transliteration, text samples contain arbitrary character stretching, irregular casing, and erratic punctuation. 

We engineered a domain-specific text normalizer:
1. **Case Normalization:** Converted all strings to lowercase to unify vocabulary.
2. **Character De-Stretching:** Normalized extended repeated characters representing elongated phonetic expressions (e.g., *"bhaaaalo"* $\to$ *"bhaalo"*, *"kooooorche"* $\to$ *"koorche"*). Regular expression used: `re.sub(r'(.)\1{2,}', r'\1\1', text)`. This preserves genuine double letters (common in Banglish, e.g., *"bhalo"* vs *"bhalo"*) while compressing emotive keyboard holding.
3. **Controlled Punctuation Retention:** Unlike standard English NLP where all punctuation is discarded, question marks (`?`) and exclamation marks (`!`) provide critical discriminative signals distinguishing interrogative customer queries (`14AC78`) from imperative jailbreak commands (`72A2C8`) and abusive exclamations (`6C0197`).
4. **Whitespace Sanitization:** Removed carriage returns, multiple spaces, and null characters.

---

## 5. Feature Engineering Techniques

### 5.1 Multi-Level N-Gram TF-IDF Feature Union
Single word tokenizers fail on Romanized Bangla because minor spelling variations generate out-of-vocabulary (OOV) tokens. To solve this, we designed a dual **Word + Subword Character N-Gram Feature Union**:

1. **Word-Level N-Grams (1 to 3 words):**
   * Captures phrase semantics and collocations (e.g., *"upeksha korun"*, *"bikretar kache"*, *"cyber akromon"*).
   * Hyperparameters: `ngram_range=(1, 3)`, `max_features=80,000`, `min_df=2`, `sublinear_tf=True`.
   * Sublinear TF scaling ($1 + \log(\text{TF})$) suppresses the dominance of high-frequency stop-words.

2. **Word-Boundary Character N-Grams (`char_wb`, 2 to 5 characters):**
   * Robust against phonetic typos, spelling variations, and morphological prefixes/suffixes (e.g., *"korte"*, *"kortam"*, *"korle"*, *"kortese"* all share character n-grams `kort`, `orte`).
   * Hyperparameters: `analyzer='char_wb'`, `ngram_range=(2, 5)`, `max_features=100,000`, `min_df=3`, `sublinear_tf=True`.

The resulting sparse feature matrix contains **180,000 engineered dimensions**, capturing both high-level semantic intent and low-level morphological root patterns.

---

## 6. Models & Experimental Results

All models were evaluated under **Stratified 5-Fold Cross-Validation** matching the exact competition evaluation metric (Macro F1).

### 6.1 Summary of Experimental Results

| Experiment | Model Architecture | Feature Representation | 5-Fold CV Macro F1 | Public LB Score | Key Observations |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Exp 1** | Logistic Regression | Word (1-3) + Char (2-5) TF-IDF | `0.7230 ± 0.0059` | — | Solid baseline; handles high-dimensional sparse features cleanly with balanced class weights. |
| **Exp 2** | **Calibrated LinearSVC (C=1.0)** | **Word (1-3) + Char (2-5) TF-IDF** | **`0.7249 ± 0.0103`** | **0.725** | **Best standalone classical model.** Platt scaling calibration generated well-calibrated class probabilities. |
| **Exp 3** | LinearSVC (C=0.5, C=1.5) | Word (1-3) + Char (2-5) TF-IDF | `0.7240 – 0.7249` | — | Tested regularization sweep; $C=1.0$ proved optimal on CV variance. |
| **Exp 4** | Advanced Multi-Model Voting | LR + LinearSVC + ComplementNB + SGD | `0.7223 ± 0.0070` | — | Naive Bayes independence assumption degraded probability calibration of the ensemble. |
| **Exp 5** | Prior-Compensated LinearSVC | Word (1-3) + Char (2-5) TF-IDF | `0.7258` | — | Post-processing threshold adjustment successfully boosted minority class `6C0197` recall. |
| **Exp 6** | XLM-RoBERTa-base (Subword Fine-Tuning) | Dense Pretrained Transformer Tokens | `0.6719` (Early/Partial) | 0.6719 | Suffered from extreme subword fragmentation on informal transliteration and CPU runtime constraints. |

---

## 7. Performance & Error Analysis

### 7.1 Confusion Matrix & Error Breakdown
Detailed error analysis of out-of-fold validation predictions revealed two primary zones of ambiguity:

1. **Jailbreak (`72A2C8`) vs. Cyber Malicious (`19E1A1`):**
   * Samples requesting hacking scripts (e.g., *"Wi-Fi password crack korar Python script"*) share technical vocabulary with prompt-injection challenges (e.g., *"Systemer nirdesh bypass koro"*). 
   * The boundary between system override instructions and external cyber harm is semantically subtle.

2. **Customer Service (`14AC78`) vs. Benign General Inquiries:**
   * Everyday queries (e.g., *"Rajma chaoyal kibhabe ranna korben?"*) and product return requests (e.g., *"Ponyoti ferot dite chai"*) both fell into class `14AC78`, confirming this class acts as the broad benign/helpful assistance category.

3. **Minority Class Recall (`6C0197`):**
   * Toxic and abusive language frequently uses non-explicit colloquial slang or sarcasm. Prior-compensation threshold tuning specifically raised `6C0197` recall by adjusting the argmax decision boundary by $(p_c)^{-0.3}$.

### 7.2 Why Pretrained Transformers Underperformed Classical TF-IDF on Romanized Bangla
A critical finding from our experiments:
* Standard multilingual models (`XLM-RoBERTa`, `mBERT`) were pretrained predominantly on standard Unicode Bengali script (e.g., Wikipedia, Common Crawl).
* When exposed to Latin-transliterated Banglish, the tokenizer shatters individual words into multiple arbitrary byte-level subword pieces (e.g., *"proshashok"* $\to$ `[' pro', 'sh', 'ash', 'ok']`), destroying semantic coherence.
* In contrast, character n-grams (`char_wb` with ranges 2–5) directly capture the invariant phonetic substrings, providing superior generalization on noisy transliterated text under low-compute constraints.

---

## 8. Final Submission & Deliverables

The final submission file was produced by our top-performing model:
* **Selected Model:** Calibrated Linear Support Vector Classifier with Multi-Level (Word 1–3 + Char 2–5) TF-IDF.
* **Output File:** `submission.csv` (and `submission_baseline.csv`)
* **Verification:** Exactly **2,048 test predictions** matching all IDs in `test.csv` with zero missing values.
* **Predicted Class Distribution:**
  * Class `14AC78`: 626 samples (30.6%)
  * Class `19E1A1`: 561 samples (27.4%)
  * Class `72A2C8`: 530 samples (25.9%)
  * Class `6C0197`: 331 samples (16.1%)

---

## 9. Repository Structure & File Descriptions

```
├── README.md                                  # Comprehensive competition report & methodology
├── Intra_2K23_NLP_BhashaQuest_Solution.ipynb # Complete annotated reproducible Jupyter Notebook
├── baseline_ml.py                             # Python pipeline for TF-IDF + LinearSVC/LR training
├── train_muril.py                             # PyTorch deep learning pipeline for MuRIL transformer
├── submission.csv                             # Final verified submission file for Kaggle
├── proba_baseline.csv                         # Out-of-fold and test class probability predictions
├── train.csv                                  # Training dataset (8,195 samples)
└── test.csv                                   # Competition test dataset (2,048 samples)
```

---

## 10. Reproduction Guide

### Environment Setup
Python 3.10+ with standard scientific libraries:
```bash
pip install scikit-learn pandas numpy scipy joblib
```
*(Optional for deep learning experiments)*:
```bash
pip install torch transformers datasets accelerate
```

### Reproducing the Winning Baseline
To train the model and generate `submission.csv`:
```bash
python baseline_ml.py
```
This runs 5-fold cross-validation, reports validation Macro F1, trains on the full dataset, and writes `submission.csv` in ~30 seconds.

---

## 11. Conclusion & Lessons Learned

Participating in the Intra 2K23 NLP Competition provided key technical insights:
1. **Linguistic Representation Beats Model Complexity:** For unstructured, low-resource Romanized text, high-dimensional character-level n-gram representations outperformed off-the-shelf multilingual transformers by resisting spelling variation and subword fragmentation.
2. **Metric-Aligned Optimization:** Standard accuracy optimization fails when evaluated on Macro F1. Platt-calibrated probability estimation combined with prior-compensated decision boundaries yielded substantial gains on the minority class.
3. **Rigorous Validation:** Stratified 5-Fold cross-validation provided an exact local proxy for leaderboard scoring, preventing overfitting and providing reproducible benchmarks.

*Special thanks to K-MINDS (Batch 2K23) for organizing an engaging and technically rigorous competition.*

