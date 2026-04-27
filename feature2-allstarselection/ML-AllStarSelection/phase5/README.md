# Phase 5 — Hindsight Model (Same-Season Stats)

## Goal

Flip the framing: use season X's own stats to predict season X's All-Stars. Answers "who DESERVED it based on their season?" rather than "who WILL make it next year?"

## Key difference from Phases 1–4

All prior phases used a lag join: season Y-1 stats → season Y All-Star label. This phase uses season Y stats → season Y All-Star label (no lag).

### Why this works better
A lag model cannot see **breakout players** — it only knows last year's stats. A player going from 6th-man-caliber to 28 PPG is invisible to it. The hindsight model sees the full current season and picks them up immediately.

### What's excluded to avoid leakage

End-of-season award shares (All-NBA, MVP, DPOY) are NOT used as features, even for the current season, because those awards are voted on *after* the All-Star game. Using them would leak future information.

Still kept (historical only): prior All-Star count, rolling averages, YoY deltas.

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9883 | 0.6847 | 0.530 | 0.968 |
| Random Forest | 0.9881 | **0.7467** | **0.631** | 0.914 |
| Gradient Boosting | 0.9863 | 0.7183 | 0.578 | 0.949 |

### Top-24 recall comparison

| Model | Phase 4 (Lag) | **Phase 5 (Hindsight)** |
|---|---|---|
| LR | 61.6% | **73.8%** |
| RF | 61.3% | **73.8%** |
| GBM | 60.7% | **72.5%** |

### 2024 demo specifically

- Phase 4 lag: 15/24 (62%)
- Phase 4 lag + availability filter: 16/24 (67%)
- **Phase 5 hindsight: 19/24 (79%)** — best single-season result

## Two valid interpretations of this model

1. **Upper bound of stats alone** — ~74% is likely the ceiling for any stats-based model. The remaining 26% is voter preference the data can't explain.

2. **"Who deserved it" analysis** — The snub list is the payoff. For 2024 the model picked Domantas Sabonis, DeMar DeRozan, and De'Aaron Fox based purely on stats; voters skipped them. That's a statistically defensible snub claim.

## Most persistent stat-based snubs

- **DeMar DeRozan** — appears as a deserved-but-not-selected player in 5+ seasons
- **Jimmy Butler** — elite advanced stats, consistently overlooked
- **Domantas Sabonis** — top-tier WS/VORP, never wins voters over
- **Chris Paul** (2017, 2018) — prime seasons skipped

## Caveat

Full-season stats include games played *after* the All-Star break. A production real-time model would need stats as-of the All-Star break (around 50 games in), which isn't in this dataset.

## Outputs

| File | What it shows |
|---|---|
| `phase5_vs_phase4.png` | Phase 4 lag model vs Phase 5 hindsight model, per-season top-24 recall |

## Run

```bash
../venv/bin/python phase5.py
```
