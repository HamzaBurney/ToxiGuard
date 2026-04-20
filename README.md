# ToxiGuard: Auditing & Mitigating Bias in Content Moderation AI

---

## Detailed Project Description

ToxiGuard is an applied machine learning project that investigates a core challenge in content moderation: building a toxicity classifier that is both effective and fair across identity groups while remaining robust against simple adversarial attacks.

The project uses the Jigsaw Unintended Bias in Toxicity Classification dataset and walks through a complete lifecycle:

1. Train a strong baseline toxicity classifier.
2. Audit that classifier for measurable group-level disparities.
3. Stress-test the model with realistic adversarial manipulations.
4. Apply and compare mitigation strategies that reduce bias while preserving utility.
5. Deploy the best model inside a practical three-layer guardrail pipeline suitable for production moderation workflows.

### Problem Context

Online moderation systems often optimize aggregate accuracy and overlook distributional harms. In practice, this can lead to uneven error rates between groups (for example, higher false positive rates on comments associated with some identities). A useful moderation model must therefore satisfy multiple objectives at once:

- maintain strong classification quality on imbalanced toxicity data,
- reduce harmful fairness gaps between cohorts,
- remain resilient to straightforward evasion attempts,
- expose uncertain cases to human reviewers instead of over-automating risky decisions.

ToxiGuard is designed to make these trade-offs explicit, measurable, and reproducible.

### Project Objectives

The project is built around five concrete objectives:

1. **Baseline performance:** Fine-tune DistilBERT for toxic vs non-toxic classification and select an operating threshold suitable for skewed label distributions.
2. **Bias measurement:** Quantify fairness gaps between a high-black identity cohort and a reference cohort using confusion-matrix-derived rates and AIF360 fairness metrics.
3. **Robustness analysis:** Demonstrate model degradation under character-level evasion and label-flipping poisoning attacks.
4. **Bias mitigation:** Compare pre-processing, post-processing, and data-level mitigation techniques under a common evaluation protocol.
5. **Deployment readiness:** Integrate the best mitigated model into a layered safety pipeline with calibration and human-in-the-loop escalation.

### End-to-End Methodology

The repository is intentionally structured as a stepwise experimental report.

- **Part 1 (Baseline):** DistilBERT is trained on stratified subsets, then evaluated with standard classification metrics and threshold analysis.
- **Part 2 (Audit):** Fairness diagnostics compare group-specific TPR/FPR/FNR and compute Statistical Parity Difference and Equal Opportunity Difference.
- **Part 3 (Adversarial):** Character perturbations and poisoning show where the classifier can be manipulated or degraded.
- **Part 4 (Mitigation):** Reweighing, threshold optimization, and oversampling are implemented and compared; the best balanced model is selected and saved.
- **Part 5 (Guardrail Pipeline):** A production-style moderation stack combines a regex safety filter, calibrated neural model, and human review queue for uncertain predictions.

### Guardrail Architecture

The final moderation design emphasizes layered risk control:

1. **Layer 1 — Input filter:** Fast regex-based blocking for severe or policy-critical patterns.
2. **Layer 2 — Calibrated model:** DistilBERT probability output is calibrated so confidence values can be operationally interpreted.
3. **Layer 3 — Human review:** Predictions in an uncertainty band are escalated to moderators.

This architecture reduces the chance that a single model output decides all outcomes, especially for borderline content.

### Key Outputs and Deliverables

By running the notebooks in order, the project produces:

- reproducible training/evaluation subsets,
- baseline and mitigated model artifacts,
- fairness and robustness analysis plots,
- threshold sensitivity analysis for review-queue design,
- an importable `ModerationPipeline` module for local inference.

### Practical Value

ToxiGuard is useful as both a learning resource and a prototype for real moderation systems. It demonstrates how to go beyond raw accuracy and build moderation that is:

- evidence-driven (metrics and diagnostics at each stage),
- fairness-aware (explicit cohort analysis and mitigation),
- robust-minded (adversarial stress testing),
- operations-ready (calibration plus human escalation).

### Scope and Limitations

This repository is an educational and experimental implementation, not a complete production service. It does not include:

- a live API or streaming moderation backend,
- persistent review tooling or annotator workflow UI,
- full policy taxonomy coverage for all abuse categories,
- continuous monitoring infrastructure for drift and fairness regression.

However, it provides a strong and extensible baseline for those production extensions.

---

## Environment

| Item | Detail |
|---|---|
| Python | 3.11.x |
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
