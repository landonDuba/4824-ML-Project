# NBA All-Star Prediction — ML Project

Predicts which NBA players will be selected as All-Stars using prior-season (and/or current-season) statistics. Built across 5 iterative phases, each adding capability and analysis over the last.

## Problem

Every NBA season, 24 players are selected as All-Stars (12 per conference). The goal is to predict those players from box-score and advanced stats.

Key challenge: **class imbalance**. Roughly 24 All-Stars out of ~450 active players per season (~1:11). Accuracy is useless — a "predict nobody" model is 94% accurate. Top-24 recall (per season) is the most meaningful metric.

---

## Project Structure

```
ML-AllStarSelection/
├── phase1/     EDA + baseline Logistic Regression
├── phase2/     Tuned LR + Random Forest, SHAP, ethics audit
├── phase3/     Enhanced features + conference-aware top-24 + Gradient Boosting
├── phase4/     Vote share features, snub analysis, availability filter, career trajectories
├── phase5/     Hindsight model (season X stats → season X All-Stars)
├── Notes.txt   Project notes
└── README.md   This file
```

Each phase folder has its own `README.md` with details, a single runnable `.py` file, and the generated charts.

---

## Results Summary

| Phase | Framing | Best Top-24 Recall | Key addition |
|---|---|---|---|
| 1 | Lag (Y-1 → Y), untuned LR baseline | 50–72% | Data pipeline + baseline |
| 2 | Lag, tuned LR + RF | ~60% | Cross-validation + SHAP |
| 3 | Lag + enhanced features | ~62% | Rolling avgs, prior AS count, conf-aware split |
| 4 | Lag + vote share + availability filter | 66% (RF, filtered) | Vote share features + snub analysis |
| 5 | Hindsight (Y → Y) | **74%** | Same-season stats (upper bound) |

---

## Models Used

- **Logistic Regression** — linear baseline, regularization tuned via cross-validation
- **Random Forest** — ensemble of decision trees, handles non-linear interactions
- **Gradient Boosting** (HistGradientBoostingClassifier) — sequential tree-boosting

All trained with time-series cross-validation so validation folds always come after training folds (no future-data leakage).

---

## Core Findings

- **VORP and Win Shares are the strongest single predictors.** Volume and impact beat efficiency.
- **Prior All-Star status** is a huge signal — voters favor established names.
- **Conference-aware prediction** (12 East + 12 West) meaningfully improves recall over global top-24.
- **The stats-only ceiling is ~74% top-24 recall.** The remaining 26% is voter preference the data can't capture without external sources (fan voting, social-media signals).
- **Persistent "stat-based snubs":** DeMar DeRozan, Jimmy Butler, Domantas Sabonis, Paul George consistently put up All-Star numbers without being selected.

---

## Dataset

Source: [NBA/ABA/BAA Stats on Kaggle](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)

Files used across phases:
- `All-Star Selections.csv` — labels
- `Player Per Game.csv` — counting stats
- `Advanced.csv` — PER, WS, BPM, VORP, etc.
- `Team Summaries.csv` — win/loss records (phase 3+)
- `Player Award Shares.csv` — MVP/DPOY voting (phase 4)
- `End of Season Teams (Voting).csv` — All-NBA voting (phase 4)

Filtered to NBA only (ABA/BAA eras excluded).

---

## How to Run

```bash
# From the project root
venv/bin/python phase1/phase1_eda.py
venv/bin/python phase2/phase2.py
venv/bin/python phase3/phase3.py
venv/bin/python phase4/phase4.py
venv/bin/python phase5/phase5.py
```

Grid-search phases (2–5) take a few minutes each.

Dependencies: `kagglehub`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`.
