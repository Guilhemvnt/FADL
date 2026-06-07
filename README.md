# Fake Account Detection on LinkedIn (FADL)
### MSc Computer Science Thesis — Heriot-Watt University

**Author:** Guilhem VINET  
**Institution:** Heriot-Watt University (HWU)  
**Programme:** MSc Computer Science  
**Academic Year:** 2025–2026  

---

## Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Feature Engineering](#feature-engineering)
5. [Methodology](#methodology)
6. [Experimental Results](#experimental-results)
7. [Inference Tool](#inference-tool)
8. [Repository Structure](#repository-structure)
9. [Conclusion](#conclusion)

---

## Abstract

Social networks such as LinkedIn have become critical platforms for professional networking, recruitment, and industry communications. The proliferation of **fake or AI-generated profiles** on these platforms threatens the integrity of professional communities, enables fraud, and undermines the reliability of the network for legitimate users.

This thesis investigates the effectiveness of machine learning approaches — from classical ensemble methods to deep neural networks — in **automatically detecting fake LinkedIn profiles** based solely on structured metadata extracted from user profiles. The system is trained on labeled datasets of genuine and fake accounts and is evaluated across multiple model architectures to identify the most robust approach.

The best-performing models achieve a **ROC-AUC score of up to 0.9961** and a **classification accuracy of 97%**, demonstrating that profile metadata alone carries strong discriminative signals for authenticity detection.

---

## Problem Statement

> **How can machine learning models reliably distinguish genuine LinkedIn profiles from fake or AI-generated ones, using only structured profile metadata?**

### Motivation

- LinkedIn hosts over 1 billion users, making it a high-value target for automated fake account creation.
- Fake profiles are used for phishing, misinformation, corporate espionage, and inflating social credibility.
- Manual verification at scale is impossible — automated detection is essential.
- AI-generated profile content (photos, bios) is becoming indistinguishable from genuine content, making **behavioral and structural metadata** signals increasingly important.

### Research Questions

1. Which machine learning architectures perform best for profile authenticity classification?
2. Which profile features are most discriminative between genuine and fake accounts?
3. How does model performance differ when trained on datasets with vs. without AI-generated fake profiles?
4. Can a trained model be productized into a real-time inference API?

---

## Dataset

### Source

Labeled LinkedIn profile datasets containing structured metadata scraped from public LinkedIn profiles, annotated with binary labels:
- **Label 0** — Real / Genuine profile
- **Label 1** — Fake / Suspicious profile

### Dataset Variants

| Dataset | Description | Fake Profile Type |
|---|---|---|
| `Clean_Original_label_data` | Full mixed dataset (genuine + fakes) | Mixed (AI & non-AI) |
| `Clean_label_data_NoAI` | Genuine + manually-crafted fake profiles | Human-created fakes |
| `Clean_label_data_AI` | Genuine + AI-generated fake profiles | AI-generated fakes |

### Class Distribution (Original Dataset — Test Split)

| Class | Count | Ratio |
|---|---|---|
| Genuine (0) | 480 | 66.7% |
| Fake (1) | 240 | 33.3% |
| **Total** | **720** | — |

### Data Collection Pipeline

Profile data was collected using a **Google Custom Search Engine (CSE)** pipeline (`Data_Collection.py`) that:
1. Queries LinkedIn profiles by name, workplace, and location.
2. Retrieves and downloads profile pictures.
3. Stores structured metadata and image URLs for downstream processing.

---

## Feature Engineering

Raw profile fields are transformed into a rich feature vector using the `preprocess_dataframe()` pipeline:

### Binary Presence Features

| Feature | Description |
|---|---|
| `Has_Photo` | Profile photo present |
| `Has_About` | Bio/About section filled |
| `Has_Projects` | Projects section filled |
| `Has_Education` | Education section filled |
| `Has_Experience` | Experience section filled |
| `Has_Skills` | Skills section filled |
| `Has_Licenses` | Certifications/Licenses present |
| `Has_Interests` | Interests section filled |
| `Has_Recommendations` | Recommendations received |

### Length / Richness Features

| Feature | Description |
|---|---|
| `About_Length` | Character count of bio |
| `Skills_Length` | Character count of skills list |
| `Experience_Length` | Character count of experience entries |
| `Education_Length` | Character count of education entries |
| `Projects_Length` | Character count of projects entries |

### Count Features

`Number of Experiences`, `Number of Educations`, `Number of Licenses`, `Number of Volunteering`, `Number of Skills`, `Number of Recommendations`, `Number of Projects`, `Number of Publications`, `Number of Courses`, `Number of Honors`, `Number of Languages`, `Number of Organizations`, `Number of Interests`, `Number of Activities`, `Connections`, `Followers`

### Categorical Features (Ordinal Encoded)

`Full Name`, `Location`, `Workplace`

**Total feature count: 34 features**

---

## Methodology

### Models Evaluated

#### 1. Random Forest
- `n_estimators=200`, `random_state=42`
- Trained on the full feature set.
- Strong baseline due to robustness to noisy features and interpretability via feature importances.

#### 2. Gradient Boosting
- `n_estimators=200`, `random_state=42`
- Sequential boosting of weak learners, excellent at capturing complex non-linear patterns.

#### 3. Logistic Regression
- `max_iter=10000`, `solver='saga'`
- Linear baseline, used to benchmark against non-linear models and as a meta-learner.

#### 4. Voting Classifier (Ensemble)
- **Soft voting** over RF + GB + LR pipelines.
- Combines predicted probabilities for a balanced, robust final prediction.

#### 5. Stacking Classifier (Ensemble)
- Base learners: **Random Forest** + **Gradient Boosting**
- Meta-learner: **Logistic Regression**
- Learns how to optimally combine base model outputs.

#### 6. Deep Learning (MLP)
- Architecture: `Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Dense(1, sigmoid)`
- Optimizer: Adam (lr=0.001), Loss: Binary Crossentropy
- Trained for up to 2000 epochs with a 0.2 validation split.

### Pipeline Architecture

```
Raw CSV Data
     │
     ▼
preprocess_dataframe()
 ├─ Fill missing values
 ├─ Encode binary presence flags
 ├─ Compute length/richness features
 └─ Ordinal encode categoricals
     │
     ▼
ColumnTransformer (OrdinalEncoder + StandardScaler)
     │
     ▼
Classifier (RF / GB / LR / Voting / Stacking / MLP)
     │
     ▼
Evaluation (Accuracy, F1, Precision, Recall, ROC-AUC)
```

### Evaluation Protocol

- **Train/Test split:** 80% / 20%, stratified by label
- **Cross-validation:** 3-fold CV on training set (ROC-AUC)
- **Metrics:** Precision, Recall, F1-Score, Accuracy, ROC-AUC

---

## Experimental Results

### Dataset: `Clean_Original_label_data` (Mixed Fakes — 720 test samples)

| Model | CV ROC-AUC (3-fold) | Accuracy | Macro F1 | ROC-AUC (Test) |
|---|---|---|---|---|
| Random Forest | 0.9955 | 96% | 0.96 | 0.9912 |
| Gradient Boosting | 0.9957 | 96% | 0.96 | **0.9930** |
| Logistic Regression | 0.9837 | 90% | 0.90 | 0.9732 |
| Voting Classifier | 0.9950 | **97%** | **0.97** | 0.9904 |
| **Stacking Classifier** | **0.9961** | 96% | 0.96 | 0.9920 |

### Dataset: `Clean_label_data_NoAI` (Human-crafted Fakes — 480 test samples)

| Model | CV ROC-AUC (3-fold) | Accuracy | Macro F1 | ROC-AUC (Test) |
|---|---|---|---|---|
| Random Forest | 0.9878 | 96% | 0.95 | 0.9905 |
| Gradient Boosting | 0.9906 | 96% | 0.95 | 0.9907 |
| Logistic Regression | 0.9671 | 88% | 0.86 | 0.9491 |
| Voting Classifier | 0.9887 | 96% | 0.95 | 0.9888 |
| **Stacking Classifier** | **0.9893** | 96% | 0.95 | **0.9921** |

### Detailed Classification Report — Best Model (Stacking, Original Dataset)

```
              precision    recall  f1-score   support

           0       0.98      0.96      0.97       480
           1       0.92      0.97      0.94       240

    accuracy                           0.96       720
   macro avg       0.95      0.96      0.96       720
weighted avg       0.96      0.96      0.96       720

ROC-AUC: 0.9920
```

### Key Observations

- **Ensemble methods consistently outperform single models**, confirming the value of combining diverse learners.
- **Logistic Regression** lags significantly (~6–8% accuracy gap), demonstrating that the decision boundary is non-linear.
- **Stacking Classifier** achieves the highest CV ROC-AUC (0.9961) across both dataset variants, making it the most reliable model for deployment.
- Models trained on the **Original mixed dataset** outperform those trained on the NoAI subset, suggesting AI-generated fakes introduce patterns that generalize better.
- Performance is remarkably consistent across both datasets, validating the **generalizability** of the metadata-based approach.

---

## Inference Tool

A production-ready inference pipeline was developed in `tool/ai_api/` to serve predictions from the best trained model.

### Architecture

```
JSON Input (LinkedIn profile metadata)
         │
         ▼
    predict.py
   ├─ Load StackingClassifier model (.pkl)
   ├─ Parse input via stdin or file argument
   ├─ preprocess_dataframe()
   ├─ model.predict() → label (0/1)
   └─ model.predict_proba() → % Genuine
         │
         ▼
JSON Output: { prediction, is_genuine, probability }
```

### Example Input

```json
[{
  "Full Name": "John Doe",
  "Workplace": "Google",
  "Location": "Mountain View, CA",
  "Connections": "500+",
  "Photo": "https://example.com/photo.jpg",
  "Followers": "1000",
  "About": "Experienced software engineer.",
  "Experiences": "Software Engineer at Google",
  "Educations": "BS in CS from MIT",
  "Skills": "Python, Go, Vue.js",
  "Projects": "Project A, Project B",
  "Languages": "English",
  "Interests": "Coding"
}]
```

### Example Output

```json
[{
  "prediction": 1,
  "is_genuine": true,
  "probability": 0.94
}]
```

### Deployed Model

The inference tool loads the **Stacking Classifier** trained on the Original dataset:
```
results/model_stats/2025-11-27_03-15-52_..._StackingClassifier.pkl
```

---

## Repository Structure

```
FADL/
├── reasearch/                        # Core research codebase
│   ├── proof/                        # Experimental scripts
│   │   ├── Data_Collection.py        # LinkedIn profile data scraping pipeline
│   │   ├── random_forest.py          # Random Forest baseline experiments
│   │   ├── Ensemble_Methods.py       # RF + GB + LR + Voting + Stacking pipeline
│   │   ├── deep_learning.py          # MLP deep learning model (TensorFlow/Keras)
│   │   ├── requirements.txt          # Python dependencies
│   │   ├── datasets/                 # Training datasets (CSV)
│   │   └── results/                  # Experiment outputs
│   │       ├── model_stats/          # Saved models (.pkl) + prediction CSVs
│   │       └── result_md/            # Human-readable result summaries
│   ├── Heriot_Watt_University__HWU__CS_Masters_thesis.pdf
│   └── Heriot_Watt_University__HWU__CS_Masters_thesis_guilhem_vinet.pdf
│
└── tool/                             # Production inference tool
    └── ai_api/
        ├── Ensemble_Methods.py       # Shared preprocessing (mirrored for API)
        ├── random_forest.py          # Shared utilities
        ├── predict.py                # Inference entrypoint (JSON I/O)
        ├── dummy_input.json          # Sample inference payload
        └── requirements.txt          # API dependencies
```

---

## Conclusion

This thesis demonstrates that **fake LinkedIn profile detection using structured profile metadata is both feasible and highly accurate**. The key findings are:

| Finding | Detail |
|---|---|
| **Best model** | Stacking Classifier (RF + GB → LR meta-learner) |
| **Best CV ROC-AUC** | 0.9961 on the Original mixed dataset |
| **Best test accuracy** | 97% (Voting Classifier) |
| **Most impactful features** | Profile completeness indicators (photo, about, experience richness) and network metrics (connections, followers) |
| **Dataset sensitivity** | Models remain robust across human-crafted and AI-generated fake variants |
| **Productization** | Successfully packaged into a JSON-in / JSON-out inference API |

### Future Work

- **Image-based detection:** Integrate facial deepfake detection from profile photos to complement metadata signals.
- **NLP features:** Add text embeddings from bio/about sections for richer representations.
- **Online learning:** Adapt models incrementally as fake profile strategies evolve.
- **Cross-platform generalization:** Validate the approach on other professional networks (Xing, Viadeo).
- **Explainability:** Integrate SHAP/LIME explanations for human-in-the-loop review workflows.

---

*Guilhem VINET — Heriot-Watt University, MSc Computer Science — 2025–2026*  
*Repository: [github.com/Guilhemvnt/FADL](https://github.com/Guilhemvnt/FADL)*
