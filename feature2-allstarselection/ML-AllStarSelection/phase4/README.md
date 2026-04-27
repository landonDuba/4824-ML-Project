# Phase 4 — Vote Share Features, Snub Analysis, Availability Filter

## Goal

Push the model's performance further using end-of-season recognition data as lag features, then dig into the results to understand who the model gets wrong and why.

## What was added over Phase 3

- **All-NBA vote share** (lagged 1 year) — strong voter-recognition signal from `End of Season Teams (Voting).csv`.
- **MVP voting share** (lagged 1 year) — star-power signal from `Player Award Shares.csv`.
- **DPOY voting share** and **All-Defense share** — defensive recognition.
- **`was_allnba`** binary flag — was the player on any All-NBA team last season?
- **Availability filter** — exclude from top-24 any player who played fewer than 30 games in the current season. Uses `current_g` joined from the label season.
- **Snub list analysis** — for each test season, print the players the model picked who weren't selected (plus the true All-Stars the model missed).
- **Player career probability trajectories** — predicted All-Star probability over the full career of 8 marquee players.

Total features: **36**.

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9734 | 0.5870 | 0.431 | 0.918 |
| Random Forest | 0.9724 | 0.6043 | 0.476 | 0.826 |
| Gradient Boosting | 0.9717 | 0.5549 | 0.402 | 0.895 |

### Availability filter impact (top-24 recall)

| Model | Unfiltered | ≥30 games | Improvement |
|---|---|---|---|
| LR | 61.6% | 64.3% | +2.6 pp |
| RF | 61.3% | **65.6%** | +4.3 pp |
| GBM | 60.7% | 64.6% | +3.9 pp |

The filter directly handles suspensions (Ja Morant) and injuries (Kawhi Leonard, Kevin Durant 2020) — cases where the model correctly scored a great player high but the player couldn't actually be an All-Star.

## Most-frequent stat-based snubs (across all test seasons)

- **DeMar DeRozan** — 4 snubs, avg prob 0.86
- **Jimmy Butler** — 4 snubs, avg prob 0.92
- **Paul George** — 4 snubs, avg prob 0.91
- **Kawhi Leonard** — 3 snubs, avg prob 0.95

## Outputs

| File | What it shows |
|---|---|
| `phase4_top24_recall.png` | Per-season top-24 recall (no filter) |
| `phase4_availability_filter.png` | RF recall before vs after the games-played filter |
| `phase4_shap.png` | Top 15 features by SHAP importance |
| `career_trajectories.png` | 8 marquee players' career probability curves, with actual All-Star seasons marked as gold stars |

## Run

```bash
../venv/bin/python phase4.py
```
