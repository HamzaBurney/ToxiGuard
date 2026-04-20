# ToxiGuard: Auditing & Mitigating Bias in Content Moderation AI

---

## Environment

| Item | Detail |
|---|---|
| Python | 3.10.x |
| GPU | NVIDIA T4 (Google Colab free tier) or local CUDA GPU |
| CUDA | 12.1 |
| Framework | HuggingFace Transformers 4.39.3, PyTorch 2.2.1 |

---

## Repository Structure

```
.
├── part1.ipynb          # Baseline DistilBERT classifier
├── part2.ipynb          # Bias audit (Black vs. White identity cohorts)
├── part3.ipynb          # Adversarial attacks (evasion + poisoning)
├── part4.ipynb          # Bias mitigation (reweighing, threshold opt, oversampling)
├── part5.ipynb          # Three-layer guardrail pipeline demonstration
├── pipeline.py          # ModerationPipeline class (importable module)
├── requirements.txt     # Pinned dependencies
└── README.md            # This file
```

**Not committed (add to .gitignore):**
```
*.csv
*.pt
*.bin
saved_model/
model_*/
results_*/
```

---

## How to Reproduce

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download

1. Create a free account at [kaggle.com](https://kaggle.com)
2. Go to: `kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification`
3. Accept the competition rules on the Data tab
4. Download **only** `jigsaw-unintended-bias-train.csv`
5. Place it in the repository root directory

### 3. Run Notebooks in Order

```bash
jupyter notebook
```

Execute notebooks **in order**:
1. `part1.ipynb` → trains baseline, saves `train_subset.csv`, `eval_subset.csv`, `eval_with_preds.csv`, `model_baseline/`
2. `part2.ipynb` → reads `eval_with_preds.csv`, loads `model_baseline/`
3. `part3.ipynb` → reads saved subsets, trains poisoned model
4. `part4.ipynb` → reads saved subsets, trains 3 mitigated models, saves `model_best_mitigated/`
5. `part5.ipynb` → loads `model_best_mitigated/`, imports `pipeline.py`

> **Tip (Colab):** Upload `jigsaw-unintended-bias-train.csv` to your Colab session storage, then run each notebook with `Runtime → Run all`. Enable GPU runtime before starting Part 1.

### 4. Smoke-Test the Pipeline

```bash
python pipeline.py
```

This runs the Layer 1 input filter on hardcoded test cases without loading the model.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| DistilBERT (not BERT-large) | Trains in 25–35 min on T4 GPU; same model family as production systems |
| Threshold = 0.4 | Better macro-F1 on imbalanced data (~8% toxic); reviewed in Part 1 |
| Stratified sampling | Preserves class balance in both train and eval splits |
| Isotonic calibration | Ensures model probabilities are meaningful for the 0.4–0.6 review band |
| Oversampling as best mitigation | Best FPR reduction on high-black cohort without catastrophic F1 loss |

---

## Fairness Impossibility Note (Part 4)

Demographic parity and equalized odds cannot be simultaneously satisfied when group base rates differ (Chouldechova 2017). The empirically-measured base rates for the high-black vs. reference cohorts differ, making the incompatibility concrete and measurable. The platform should choose equalized FPR as the primary fairness target.
