# FADL — Inference Tool

This folder contains the **production inference tool** for the Fake Account Detection on LinkedIn (FADL) project.  
It loads the best trained model (Stacking Classifier) and predicts whether a LinkedIn profile is genuine or fake from structured metadata.

---

## Prerequisites

- **Python 3.10+**
- A trained model `.pkl` file in `ai_api/results/model_stats/` (see [Model](#model) below)

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r ai_api/requirements.txt
```

> **Note:** The `requirements.txt` is a full pinned lockfile. If you only need the inference tool (no training), you can install the minimal set instead:
> ```bash
> pip install pandas scikit-learn joblib
> ```

---

## Model

The inference script expects a trained **Stacking Classifier** `.pkl` file at the following path (relative to `ai_api/`):

```
ai_api/results/model_stats/2025-11-27_03-15-52_LinkedIn people profiles datasets - Clean_Original_label_data_StackingClassifier.pkl
```

If you need to retrain the model, run the ensemble training script from the `reasearch/proof/` directory (see the [research README](../reasearch/READMED.md)).

---

## Running the Inference

Navigate into the `ai_api/` folder first:

```bash
cd ai_api/
```

### Option 1 — From a JSON file (recommended)

Pass the path to a JSON file as an argument:

```bash
python predict.py dummy_input.json
```

### Option 2 — With your own input file

Create a JSON file following the schema below, then run:

```bash
python predict.py /path/to/your/profile.json
```

---

## Input Format

The input must be a **JSON array** of profile objects. Each object should contain the following fields:

```json
[
  {
    "Full Name": "Jane Smith",
    "Workplace": "Acme Corp",
    "Location": "Paris, France",
    "Connections": "300",
    "Photo": "https://example.com/photo.jpg",
    "Followers": "450",
    "About": "Product manager with 8 years of experience.",
    "Experiences": "Product Manager at Acme Corp",
    "Educations": "MSc Management, HEC Paris",
    "Licenses": "",
    "Volunteering": "",
    "Skills": "Product Strategy, Agile, SQL",
    "Recommendations": "2 recommendations",
    "Projects": "",
    "Publications": "",
    "Courses": "",
    "Honors": "",
    "Scores": "",
    "Languages": "French, English",
    "Organizations": "",
    "Interests": "Tech, Innovation",
    "Activities": ""
  }
]
```

> **Tip:** A working example is already provided in `ai_api/dummy_input.json`.

---

## Output Format

The script prints a JSON array to stdout, one result per input profile:

```json
[
  {
    "prediction": 1,
    "is_genuine": true,
    "probability": 0.94
  }
]
```

| Field | Type | Description |
|---|---|---|
| `prediction` | `int` | `1` = Genuine, `0` = Fake |
| `is_genuine` | `bool` | `true` if the model predicts a genuine profile |
| `probability` | `float` | Confidence score for the Genuine class (0.0 – 1.0) |

---

## File Structure

```
tool/
├── README.md               ← You are here
└── ai_api/
    ├── predict.py          ← Main inference script
    ├── Ensemble_Methods.py ← Preprocessing & model utilities
    ├── random_forest.py    ← Shared feature helpers
    ├── dummy_input.json    ← Sample input payload
    ├── requirements.txt    ← Python dependencies
    └── results/
        └── model_stats/    ← Trained model .pkl files go here
```
