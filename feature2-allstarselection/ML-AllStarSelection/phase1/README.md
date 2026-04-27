# Phase 1 — Exploratory Data Analysis & Baseline

## Goal

Understand the data and prove there's signal worth modeling before writing real algorithms.

## What was done

- **Built the data pipeline** — the NBA stats dataset has a dedicated `All-Star Selections.csv`. Created a lag-join: take a player's stats from season Y-1 and label them by whether they were an All-Star in season Y.
- **Handled traded players** — players traded mid-season have both individual-team rows and a `TOT` total row. Kept only the `TOT` row to avoid double-counting.
- **Measured class imbalance** — about 24 All-Stars per 450 active players = 1:11 imbalance. Accuracy is useless for this problem; need F1 and top-24 recall.
- **Ranked features by correlation with the All-Star label** — VORP (0.58) and Win Shares (0.54) are strongest. Volume and impact matter more than efficiency.
- **Ran a throwaway baseline** — untuned Logistic Regression with `class_weight='balanced'`.

## Results

- **ROC-AUC: 0.97** — strong signal in the data
- **Top-24 recall per season: 50–72%** — real room to improve
- **Precision: 0.40** — too many false positives

## Outputs

| File | What it shows |
|---|---|
| `feature_correlation.png` | Bar chart of each feature's correlation with the All-Star label |
| `allstar_rate_by_season.png` | All-Star selection rate over the history of the NBA |
| `baseline_confusion_matrix.png` | Phase 1 baseline LR confusion matrix with F1 score in title |
| `roc_curve.png` | ROC curve with AUC |

## Run

```bash
../venv/bin/python phase1_eda.py
```
