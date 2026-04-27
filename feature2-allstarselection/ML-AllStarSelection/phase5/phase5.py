"""
Phase 5 — Hindsight Model
Uses season X's own stats to predict season X's All-Stars.
This is "who DESERVED it based on their season" rather than "who will make it".

Note: this is hindsight because full-season stats include post-All-Star-game
performance. End-of-season award shares (All-NBA, MVP, DPOY) are excluded
because those are determined after the All-Star game (data leakage).
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# ── 1. Load data ───────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("sumitrodatta/nba-aba-baa-stats")

allstar      = pd.read_csv(os.path.join(path, "All-Star Selections.csv"))
per_game     = pd.read_csv(os.path.join(path, "Player Per Game.csv"))
advanced     = pd.read_csv(os.path.join(path, "Advanced.csv"))
team_summary = pd.read_csv(os.path.join(path, "Team Summaries.csv"))

allstar_nba   = allstar[allstar['lg'] == 'NBA'].copy()
allstar_flags = (
    allstar_nba[['player_id', 'season']]
    .drop_duplicates()
    .assign(all_star=1)
)

def deduplicate_players(df):
    multi  = df[df.duplicated(subset=['player_id', 'season'], keep=False)]
    tot    = multi[multi['team'] == 'TOT']
    single = df[~df.duplicated(subset=['player_id', 'season'], keep=False)]
    return pd.concat([single, tot], ignore_index=True)

per_game_clean = deduplicate_players(per_game[per_game['lg'] == 'NBA'])
advanced_clean = deduplicate_players(advanced[advanced['lg'] == 'NBA'])

stats = per_game_clean.merge(
    advanced_clean[['player_id', 'season', 'per', 'ts_percent', 'ws', 'ws_48',
                    'bpm', 'vorp', 'usg_percent', 'obpm', 'dbpm']],
    on=['player_id', 'season'],
    how='inner'
).sort_values(['player_id', 'season'])

# ── 2. Team win% ───────────────────────────────────────────────────────────────
win_pct = team_summary[team_summary['lg'] == 'NBA'][['season', 'abbreviation', 'w', 'l']].copy()
win_pct['win_pct'] = win_pct['w'] / (win_pct['w'] + win_pct['l'])
stats = stats.merge(
    win_pct[['season', 'abbreviation', 'win_pct']].rename(columns={'abbreviation': 'team'}),
    on=['season', 'team'], how='left'
)
stats['win_pct'] = stats['win_pct'].fillna(0.5)

# ── 3. Rolling averages and YoY deltas (historical context) ──────────────────
roll_cols  = ['pts_per_game', 'ast_per_game', 'trb_per_game', 'ws', 'vorp', 'bpm']
delta_cols = ['pts_per_game', 'ws', 'vorp', 'bpm']

for col in roll_cols:
    stats[f'{col}_2yr'] = (
        stats.groupby('player_id')[col]
        .transform(lambda x: x.rolling(2, min_periods=1).mean())
    )
for col in delta_cols:
    stats[f'{col}_delta'] = (
        stats.groupby('player_id')[col]
        .transform(lambda x: x.diff().fillna(0))
    )

# ── 4. Same-season join (NO LAG — hindsight) ──────────────────────────────────
# Label the player's All-Star status in the SAME season as their stats
merged = stats.merge(
    allstar_flags,
    on=['player_id', 'season'],
    how='left'
)
merged['all_star'] = merged['all_star'].fillna(0).astype(int)

valid_seasons = allstar_flags['season'].unique()
merged = merged[merged['season'].isin(valid_seasons)].sort_values('season')
merged = merged.rename(columns={'season': 'label_season'})

# ── 5. Prior All-Star count (historical — no leakage) ─────────────────────────
for lag in [1, 2, 3]:
    temp = allstar_flags[['player_id', 'season']].copy()
    temp['label_season'] = temp['season'] + lag
    temp[f'was_as_{lag}'] = 1
    merged = merged.merge(temp[['player_id', 'label_season', f'was_as_{lag}']],
                          on=['player_id', 'label_season'], how='left')
    merged[f'was_as_{lag}'] = merged[f'was_as_{lag}'].fillna(0).astype(int)
merged['prior_as_3yr'] = merged['was_as_1'] + merged['was_as_2'] + merged['was_as_3']

# ── 6. Conference ─────────────────────────────────────────────────────────────
EAST = {'ATL','BOS','BRK','CHI','CHO','CLE','DET','IND','MIA','MIL',
        'NJN','NYK','ORL','PHI','TOR','WAS','CHH','CHA'}
WEST = {'DAL','DEN','GSW','HOU','LAC','LAL','MEM','MIN','NOP','OKC',
        'PHO','POR','SAC','SAS','UTA','SEA','NOH','NOK','VAN'}
merged['conference'] = merged['team'].apply(
    lambda t: 'East' if t in EAST else ('West' if t in WEST else 'Unknown')
)

# ── 7. Feature set — NO end-of-season awards (would leak) ────────────────────
BASE_COLS = [
    'pts_per_game', 'ast_per_game', 'trb_per_game', 'stl_per_game', 'blk_per_game',
    'fg_percent', 'ft_percent', 'mp_per_game', 'g',
    'per', 'ts_percent', 'ws', 'ws_48', 'bpm', 'vorp', 'usg_percent'
]
HIST_COLS  = ['prior_as_3yr', 'was_as_1', 'was_as_2', 'was_as_3']
ROLL_COLS  = [f'{c}_2yr'   for c in roll_cols]
DELTA_COLS = [f'{c}_delta' for c in delta_cols]
OTHER_COLS = ['win_pct']

FEATURE_COLS = BASE_COLS + HIST_COLS + ROLL_COLS + DELTA_COLS + OTHER_COLS
print(f"Total features: {len(FEATURE_COLS)}")
print(f"Framing: season X stats → season X All-Star label (hindsight)\n")

# ── 8. Train / test split ─────────────────────────────────────────────────────
train = merged[merged['label_season'] < 2015].copy()
test  = merged[merged['label_season'] >= 2015].copy()

X_train = train[FEATURE_COLS].fillna(0)
y_train = train['all_star']
X_test  = test[FEATURE_COLS].fillna(0)
y_test  = test['all_star']

print(f"Train: {len(train)} rows | {y_train.sum()} All-Stars")
print(f"Test : {len(test)} rows  | {y_test.sum()} All-Stars\n")

tscv = TimeSeriesSplit(n_splits=5)

# ── 9. Model training ─────────────────────────────────────────────────────────
print("=== Tuning Logistic Regression ===")
lr_cv = GridSearchCV(
    Pipeline([('scaler', StandardScaler()),
              ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))]),
    {'clf__C': [0.01, 0.1, 1.0, 10.0]},
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
lr_cv.fit(X_train, y_train)
best_lr  = lr_cv.best_estimator_
lr_proba = best_lr.predict_proba(X_test)[:, 1]
lr_pred  = best_lr.predict(X_test)
print(f"Best C: {lr_cv.best_params_['clf__C']}  |  CV AUC: {lr_cv.best_score_:.4f}")

print("\n=== Tuning Random Forest ===")
rf_cv = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {'n_estimators': [200, 400], 'max_depth': [6, 10, None], 'min_samples_leaf': [1, 5, 10]},
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
rf_cv.fit(X_train, y_train)
best_rf  = rf_cv.best_estimator_
rf_proba = best_rf.predict_proba(X_test)[:, 1]
rf_pred  = best_rf.predict(X_test)
print(f"Best params: {rf_cv.best_params_}  |  CV AUC: {rf_cv.best_score_:.4f}")

print("\n=== Tuning Gradient Boosting ===")
sw = y_train.map({0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()})
hgb_cv = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    {'max_iter': [300, 500], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]},
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
hgb_cv.fit(X_train, y_train, sample_weight=sw)
best_hgb  = hgb_cv.best_estimator_
hgb_proba = best_hgb.predict_proba(X_test)[:, 1]
hgb_pred  = best_hgb.predict(X_test)
print(f"Best params: {hgb_cv.best_params_}  |  CV AUC: {hgb_cv.best_score_:.4f}")

# ── 10. Head-to-head ──────────────────────────────────────────────────────────
print("\n=== Standard Metrics — Hindsight Model ===")
def evaluate(name, y_true, y_pred, y_proba):
    return {
        'Model':     name,
        'ROC-AUC':   round(roc_auc_score(y_true, y_proba), 4),
        'F1':        round(f1_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
    }
results = pd.DataFrame([
    evaluate('LR',  y_test, lr_pred,  lr_proba),
    evaluate('RF',  y_test, rf_pred,  rf_proba),
    evaluate('GBM', y_test, hgb_pred, hgb_proba),
])
print(results.to_string(index=False))

print("\nPhase 4 (lag model) reference:")
print("  LR  AUC 0.9734  F1 0.5870  Precision 0.4314  Recall 0.9180")
print("  RF  AUC 0.9724  F1 0.6043  Precision 0.4764  Recall 0.8262")
print("  GBM AUC 0.9717  F1 0.5549  Precision 0.4021  Recall 0.8951")

# ── 11. Conference-aware top-24 ───────────────────────────────────────────────
test_eval = test.copy()
test_eval['lr_proba']  = lr_proba
test_eval['rf_proba']  = rf_proba
test_eval['hgb_proba'] = hgb_proba

print("\n=== Conference-Aware Top-24 Recall (Hindsight) ===")
season_rows = []
for season, grp in test_eval.groupby('label_season'):
    total = grp['all_star'].sum()
    hits = {m: 0 for m in ['lr', 'rf', 'hgb']}
    for conf, slots in [('East', 12), ('West', 12)]:
        cg = grp[grp['conference'] == conf]
        if cg.empty:
            continue
        for m in hits:
            hits[m] += cg.nlargest(slots, f'{m}_proba')['all_star'].sum()
    season_rows.append({'season': season, 'true_AS': total, **hits})
    print(f"  {season}: "
          f"LR {hits['lr']}/{total} ({hits['lr']/total*100:.0f}%)  |  "
          f"RF {hits['rf']}/{total} ({hits['rf']/total*100:.0f}%)  |  "
          f"GBM {hits['hgb']}/{total} ({hits['hgb']/total*100:.0f}%)")

season_df = pd.DataFrame(season_rows)
print()
for m, label in [('lr','LR'),('rf','RF'),('hgb','GBM')]:
    avg = season_df[m].sum() / season_df['true_AS'].sum()
    print(f"{label} overall recall: {avg:.1%}")
print("\nPhase 4 lag model overall recall (unfiltered): LR 61.6%  RF 61.3%  GBM 60.7%")
print("Phase 4 lag model overall recall (filtered):    LR 64.3%  RF 65.6%  GBM 64.6%")

# ── 12. Side-by-side comparison chart ─────────────────────────────────────────
# Phase 4 numbers by season (unfiltered RF from phase4 output)
phase4_rf = {2015:18, 2016:19, 2017:18, 2018:16, 2019:18, 2020:13,
             2021:16, 2022:13, 2023:13, 2024:15, 2025:13, 2026:15}
phase5_rf = dict(zip(season_df['season'], season_df['rf']))

seasons_x = sorted(phase4_rf.keys())
p4_vals = [phase4_rf[s] for s in seasons_x]
p5_vals = [phase5_rf.get(s, 0) for s in seasons_x]

x = np.arange(len(seasons_x))
w = 0.35
plt.figure(figsize=(13, 5))
plt.bar(x - w/2, p4_vals, w, label='Phase 4 (lag: stats Y-1 → AS Y)',     color='steelblue')
plt.bar(x + w/2, p5_vals, w, label='Phase 5 (hindsight: stats Y → AS Y)', color='seagreen')
plt.xticks(x, seasons_x, rotation=45)
plt.ylabel('True All-Stars in Top 24 (conf-aware)')
plt.title('Lag Model vs Hindsight Model — RF Top-24 Recall')
plt.legend()
plt.tight_layout()
plt.savefig('phase5_vs_phase4.png', dpi=120)
plt.close()
print("\nSaved: phase5_vs_phase4.png")

# ── 13. "Who deserved it?" — stat-based snubs ────────────────────────────────
print("\n" + "="*60)
print("STAT-BASED SNUBS — Who deserved it based on their season?")
print("="*60)
print("(Players the hindsight model put in top-24 but voters didn't select)\n")

name_col = 'player' if 'player' in test_eval.columns else 'player_id'

for season, grp in test_eval.groupby('label_season'):
    selected_idx = []
    for conf, slots in [('East', 12), ('West', 12)]:
        cg = grp[grp['conference'] == conf]
        selected_idx += cg.nlargest(slots, 'rf_proba').index.tolist()

    snubbed = grp.loc[
        grp.index.isin(selected_idx) & (grp['all_star'] == 0),
        [name_col, 'team', 'rf_proba', 'pts_per_game', 'ws', 'vorp', 'g']
    ].sort_values('rf_proba', ascending=False)

    if not snubbed.empty:
        print(f"── {season} (deserved by stats, not selected) ──")
        for _, r in snubbed.iterrows():
            print(f"  {r[name_col]:<26} {r['team']}  "
                  f"PTS={r['pts_per_game']:.1f}  WS={r['ws']:.1f}  "
                  f"VORP={r['vorp']:.1f}  G={int(r['g'])}  prob={r['rf_proba']:.3f}")
        print()

# ── 14. Demo ──────────────────────────────────────────────────────────────────
def predict_allstars_hindsight(year: int, model=best_rf):
    candidates = merged[merged['label_season'] == year].copy()
    if candidates.empty:
        print(f"No data for {year}.")
        return None
    name_col = 'player' if 'player' in candidates.columns else 'player_id'
    candidates['probability'] = model.predict_proba(candidates[FEATURE_COLS].fillna(0))[:, 1]
    parts = []
    for conf, slots in [('East', 12), ('West', 12)]:
        parts.append(candidates[candidates['conference'] == conf].nlargest(slots, 'probability'))
    result = (
        pd.concat(parts)
        .sort_values('probability', ascending=False)
        [[name_col, 'label_season', 'team', 'conference', 'probability',
          'pts_per_game', 'ws', 'vorp', 'all_star']]
        .rename(columns={'label_season': 'season', 'all_star': 'actual_allstar'})
        .reset_index(drop=True)
    )
    result.index += 1
    return result

print("\n=== Hindsight Demo: Top-24 Stat-Based All-Stars for 2024 ===")
demo = predict_allstars_hindsight(2024)
if demo is not None:
    print(demo.round(3).to_string())
    correct = demo['actual_allstar'].sum()
    print(f"\n{correct}/24 matched actual All-Stars ({correct/24*100:.0f}% top-24 recall)")
    print(f"Phase 4 lag model for 2024: 15–16 / 24")
