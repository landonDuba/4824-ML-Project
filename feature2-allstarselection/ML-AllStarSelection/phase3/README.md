# Phase 3 — Enhanced Features & Conference-Aware Evaluation

## Goal

Improve on Phase 2 by adding context features the model didn't have, switching to conference-aware selection (matching real NBA rules), and benchmarking a third model.

## What was added over Phase 2

- **Team win %** from `Team Summaries.csv` — voters favor players on winning teams.
- **Prior All-Star count** — was the player an All-Star in each of the last 3 seasons? Strong voter-familiarity signal.
- **2-year rolling averages** for pts, ast, trb, ws, vorp, bpm — smooths out injury-shortened seasons.
- **Year-over-year deltas** for pts, ws, vorp, bpm — captures trajectory (improving vs declining).
- **HistGradientBoostingClassifier** (sklearn's equivalent of XGBoost) added as a third model.
- **Conference-aware top-24** — instead of picking top 24 globally, pick top 12 East + top 12 West separately. This matches how All-Star teams are actually constructed.

Total features: **31** (up from 16 in Phase 2).

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9730 | 0.5809 | 0.425 | 0.918 |
| Random Forest | 0.9716 | 0.6002 | 0.470 | 0.830 |
| Gradient Boosting | 0.9711 | 0.5515 | 0.399 | 0.895 |

- **Top-24 conf-aware recall: 62.0% (LR), 60.3% (RF), 61.0% (GBM)**
- Modest but consistent improvement over Phase 2 on AUC, F1, and precision.

## Key SHAP finding

The new features work. The `pts_per_game_2yr`, `ws_2yr`, `prior_as_3yr`, and `was_as_1` features all land in the top 10 by SHAP importance — confirming that rolling averages and All-Star history are genuinely predictive.

## Outputs

| File | What it shows |
|---|---|
| `phase3_top24_recall.png` | Per-season top-24 conf-aware recall for all 3 models |
| `phase3_roc.png` | ROC curves for LR, RF, GBM |
| `phase3_shap.png` | Top 15 features by mean |SHAP| |

## Run

```bash
../venv/bin/python phase3.py
```
