"""
Phase 4 — Vote Share Features + Deep Analysis
New features: All-NBA vote share, MVP vote share, DPOY vote share (all lagged 1 year)
New analysis: snub list per season, player career probability trajectories
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import shap

# ── 1. Load data ───────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("sumitrodatta/nba-aba-baa-stats")

allstar      = pd.read_csv(os.path.join(path, "All-Star Selections.csv"))
per_game     = pd.read_csv(os.path.join(path, "Player Per Game.csv"))
advanced     = pd.read_csv(os.path.join(path, "Advanced.csv"))
team_summary = pd.read_csv(os.path.join(path, "Team Summaries.csv"))
awards       = pd.read_csv(os.path.join(path, "Player Award Shares.csv"))
eostv        = pd.read_csv(os.path.join(path, "End of Season Teams (Voting).csv"))

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
stats = stats.merge(win_pct[['season', 'abbreviation', 'win_pct']].rename(columns={'abbreviation': 'team'}),
                    on=['season', 'team'], how='left')
stats['win_pct'] = stats['win_pct'].fillna(0.5)

# ── 3. Vote share features (same-season — will be lagged via the lag join) ────
# All-NBA vote share: take the max share per player-season (handles duplicate rows)
allnba = (
    eostv[eostv['type'] == 'all_nba']
    .groupby(['player_id', 'season'])['share']
    .max()
    .reset_index()
    .rename(columns={'share': 'allnba_share'})
)
# All-Defense vote share
alldef = (
    eostv[eostv['type'] == 'all_defense']
    .groupby(['player_id', 'season'])['share']
    .max()
    .reset_index()
    .rename(columns={'share': 'alldef_share'})
)
# MVP vote share
mvp_vs = (
    awards[awards['award'] == 'nba mvp']
    [['player_id', 'season', 'share']]
    .rename(columns={'share': 'mvp_share'})
)
# DPOY vote share
dpoy_vs = (
    awards[awards['award'] == 'nba dpoy']
    [['player_id', 'season', 'share']]
    .rename(columns={'share': 'dpoy_share'})
)

for df, col in [(allnba, 'allnba_share'), (alldef, 'alldef_share'),
                (mvp_vs, 'mvp_share'), (dpoy_vs, 'dpoy_share')]:
    stats = stats.merge(df, on=['player_id', 'season'], how='left')
    stats[col] = stats[col].fillna(0)

# Was this player on any All-NBA team this season?
stats['was_allnba'] = (stats['allnba_share'] > 0).astype(int)

# ── 4. Rolling averages and YoY deltas ────────────────────────────────────────
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

# ── 5. Lag join ───────────────────────────────────────────────────────────────
stats_lag = stats.copy()
stats_lag['label_season'] = stats_lag['season'] + 1

merged = stats_lag.merge(
    allstar_flags.rename(columns={'season': 'label_season'}),
    on=['player_id', 'label_season'],
    how='left'
)
merged['all_star'] = merged['all_star'].fillna(0).astype(int)
valid_seasons = allstar_flags['season'].unique()
merged = merged[merged['label_season'].isin(valid_seasons)].sort_values('label_season')

# Join current-season (label_season) games played as an availability signal.
# At All-Star selection time (February), voters know who's been playing.
current_g = per_game_clean[['player_id', 'season', 'g']].rename(
    columns={'season': 'label_season', 'g': 'current_g'}
)
current_g = current_g.groupby(['player_id', 'label_season'])['current_g'].max().reset_index()
merged = merged.merge(current_g, on=['player_id', 'label_season'], how='left')
merged['current_g'] = merged['current_g'].fillna(0)

# ── 6. Prior All-Star count ───────────────────────────────────────────────────
for lag in [1, 2, 3]:
    temp = allstar_flags[['player_id', 'season']].copy()
    temp['label_season'] = temp['season'] + lag
    temp[f'was_as_{lag}'] = 1
    merged = merged.merge(temp[['player_id', 'label_season', f'was_as_{lag}']],
                          on=['player_id', 'label_season'], how='left')
    merged[f'was_as_{lag}'] = merged[f'was_as_{lag}'].fillna(0).astype(int)
merged['prior_as_3yr'] = merged['was_as_1'] + merged['was_as_2'] + merged['was_as_3']

# ── 7. Conference ─────────────────────────────────────────────────────────────
EAST = {'ATL','BOS','BRK','CHI','CHO','CLE','DET','IND','MIA','MIL',
        'NJN','NYK','ORL','PHI','TOR','WAS','CHH','CHA'}
WEST = {'DAL','DEN','GSW','HOU','LAC','LAL','MEM','MIN','NOP','OKC',
        'PHO','POR','SAC','SAS','UTA','SEA','NOH','NOK','VAN'}
merged['conference'] = merged['team'].apply(
    lambda t: 'East' if t in EAST else ('West' if t in WEST else 'Unknown')
)

# ── 8. Feature set ─────────────────────────────────────────────────────────────
BASE_COLS = [
    'pts_per_game', 'ast_per_game', 'trb_per_game', 'stl_per_game', 'blk_per_game',
    'fg_percent', 'ft_percent', 'mp_per_game', 'g',
    'per', 'ts_percent', 'ws', 'ws_48', 'bpm', 'vorp', 'usg_percent'
]
VOTE_COLS  = ['allnba_share', 'alldef_share', 'mvp_share', 'dpoy_share', 'was_allnba']
HIST_COLS  = ['prior_as_3yr', 'was_as_1', 'was_as_2', 'was_as_3']
ROLL_COLS  = [f'{c}_2yr'   for c in roll_cols]
DELTA_COLS = [f'{c}_delta' for c in delta_cols]
OTHER_COLS = ['win_pct']

FEATURE_COLS = BASE_COLS + VOTE_COLS + HIST_COLS + ROLL_COLS + DELTA_COLS + OTHER_COLS
print(f"Total features: {len(FEATURE_COLS)}  (+{len(VOTE_COLS)} vote share features vs phase3)")

# ── 9. Train / test split ─────────────────────────────────────────────────────
train = merged[merged['label_season'] < 2015].copy()
test  = merged[merged['label_season'] >= 2015].copy()

X_train = train[FEATURE_COLS].fillna(0)
y_train = train['all_star']
X_test  = test[FEATURE_COLS].fillna(0)
y_test  = test['all_star']

print(f"Train: {len(train)} rows | {y_train.sum()} All-Stars")
print(f"Test : {len(test)} rows  | {y_test.sum()} All-Stars\n")

tscv = TimeSeriesSplit(n_splits=5)

# ── 10. Model training ────────────────────────────────────────────────────────
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

# ── 11. Head-to-head + conference-aware top-24 ────────────────────────────────
print("\n=== Standard Metrics ===")
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

test_eval = test.copy()
test_eval['lr_proba']  = lr_proba
test_eval['rf_proba']  = rf_proba
test_eval['hgb_proba'] = hgb_proba

print("\n=== Conference-Aware Top-24 Recall ===")
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
    row = {'season': season, 'true_AS': total, **hits}
    season_rows.append(row)
    print(f"  {season}: "
          f"LR {hits['lr']}/{total} ({hits['lr']/total*100:.0f}%)  |  "
          f"RF {hits['rf']}/{total} ({hits['rf']/total*100:.0f}%)  |  "
          f"GBM {hits['hgb']}/{total} ({hits['hgb']/total*100:.0f}%)")

season_df = pd.DataFrame(season_rows)
for m, label in [('lr','LR'),('rf','RF'),('hgb','GBM')]:
    avg = season_df[m].sum() / season_df['true_AS'].sum()
    print(f"{label} overall recall: {avg:.1%}")

# ── 11b. Availability filter: exclude players who played < 30 games ──────────
MIN_GAMES = 30
print(f"\n=== Conference-Aware Top-24 WITH Availability Filter (≥{MIN_GAMES} games played) ===")

filter_rows = []
for season, grp in test_eval.groupby('label_season'):
    total = grp['all_star'].sum()
    eligible = grp[grp['current_g'] >= MIN_GAMES]
    hits = {m: 0 for m in ['lr', 'rf', 'hgb']}
    for conf, slots in [('East', 12), ('West', 12)]:
        cg = eligible[eligible['conference'] == conf]
        if cg.empty:
            continue
        for m in hits:
            hits[m] += cg.nlargest(slots, f'{m}_proba')['all_star'].sum()
    removed = len(grp) - len(eligible)
    filter_rows.append({'season': season, 'true_AS': total, 'removed': removed, **hits})
    print(f"  {season}: removed {removed:>3} ineligible  |  "
          f"LR {hits['lr']}/{total} ({hits['lr']/total*100:.0f}%)  |  "
          f"RF {hits['rf']}/{total} ({hits['rf']/total*100:.0f}%)  |  "
          f"GBM {hits['hgb']}/{total} ({hits['hgb']/total*100:.0f}%)")

filter_df = pd.DataFrame(filter_rows)
print()
for m, label in [('lr','LR'),('rf','RF'),('hgb','GBM')]:
    unfiltered = season_df[m].sum() / season_df['true_AS'].sum()
    filtered   = filter_df[m].sum() / filter_df['true_AS'].sum()
    delta_pp   = (filtered - unfiltered) * 100
    print(f"{label} recall — unfiltered: {unfiltered:.1%}  |  filtered: {filtered:.1%}  |  Δ {delta_pp:+.1f} pp")

# Side-by-side comparison chart
x = np.arange(len(season_df))
w = 0.35
plt.figure(figsize=(13, 5))
plt.bar(x-w/2, season_df['rf'], w, label=f'RF (no filter)',         color='darkorange', alpha=0.6)
plt.bar(x+w/2, filter_df['rf'], w, label=f'RF (≥{MIN_GAMES} games)', color='darkorange')
plt.xticks(x, season_df['season'].astype(int), rotation=45)
plt.ylabel('True All-Stars in Top 24')
plt.title('Availability Filter Impact — Random Forest Top-24 Recall per Season')
plt.legend()
plt.tight_layout()
plt.savefig('phase4_availability_filter.png', dpi=120)
plt.close()
print("\nSaved: phase4_availability_filter.png")

# Bar chart
x, w = np.arange(len(season_df)), 0.25
plt.figure(figsize=(13, 5))
plt.bar(x-w,   season_df['lr'],  w, label='LR',  color='steelblue')
plt.bar(x,     season_df['rf'],  w, label='RF',  color='darkorange')
plt.bar(x+w,   season_df['hgb'],w, label='GBM', color='seagreen')
plt.xticks(x, season_df['season'].astype(int), rotation=45)
plt.ylabel('True All-Stars in Top 24 (conf-aware)')
plt.title('Phase 4 — Conference-Aware Top-24 Recall per Season')
plt.legend()
plt.tight_layout()
plt.savefig('phase4_top24_recall.png', dpi=120)
plt.close()
print("\nSaved: phase4_top24_recall.png")

# ── 12. SHAP — what do the vote share features contribute? ───────────────────
print("\n=== SHAP Feature Attribution (RF) ===")
explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
print("\nTop 15 features by SHAP importance:")
print(mean_shap.head(15).round(4).to_string())

plt.figure(figsize=(8, 7))
mean_shap.head(15).plot(kind='barh', color='darkorange')
plt.gca().invert_yaxis()
plt.xlabel('Mean |SHAP value|')
plt.title('Feature Importance — Phase 4 (RF with vote share features)')
plt.tight_layout()
plt.savefig('phase4_shap.png', dpi=120)
plt.close()
print("Saved: phase4_shap.png")

# ── 13. SNUB LIST analysis ────────────────────────────────────────────────────
print("\n" + "="*60)
print("SNUB LIST — High probability, not selected (RF model)")
print("="*60)

# Use RF as primary model for analysis
test_eval['rf_rank'] = test_eval.groupby('label_season')['rf_proba'].rank(ascending=False)

snubs_all = []
for season, grp in test_eval.groupby('label_season'):
    # Conference-aware top 24: top 12 East + top 12 West
    selected_idx = []
    for conf, slots in [('East', 12), ('West', 12)]:
        cg = grp[grp['conference'] == conf]
        selected_idx += cg.nlargest(slots, 'rf_proba').index.tolist()

    predicted_set = set(selected_idx)
    name_col = 'player' if 'player' in grp.columns else 'player_id'

    # Snubs: in predicted top-24, but not actual All-Star
    snubbed = grp.loc[
        grp.index.isin(predicted_set) & (grp['all_star'] == 0),
        [name_col, 'team', 'conference', 'rf_proba', 'allnba_share', 'mvp_share', 'all_star']
    ].copy()
    snubbed['season'] = season
    snubs_all.append(snubbed)

    # Missed: actual All-Star NOT in predicted top-24
    missed = grp.loc[
        ~grp.index.isin(predicted_set) & (grp['all_star'] == 1),
        [name_col, 'team', 'conference', 'rf_proba', 'all_star']
    ].copy()

    if not snubbed.empty or not missed.empty:
        print(f"\n── {season} ──")
        if not snubbed.empty:
            print(f"  Model predicted but NOT selected ({len(snubbed)}):")
            for _, r in snubbed.sort_values('rf_proba', ascending=False).iterrows():
                print(f"    {r[name_col]:<28} {r['team']}  prob={r['rf_proba']:.3f}  "
                      f"allnba_share={r['allnba_share']:.2f}  mvp_share={r['mvp_share']:.2f}")
        if not missed.empty:
            print(f"  Actual All-Stars model MISSED ({len(missed)}):")
            for _, r in missed.sort_values('rf_proba', ascending=False).iterrows():
                print(f"    {r[name_col]:<28} {r['team']}  prob={r['rf_proba']:.3f}")

snubs_df = pd.concat(snubs_all, ignore_index=True)
name_col = 'player' if 'player' in snubs_df.columns else 'player_id'

# Most frequent snubs across all seasons
print("\n=== Most Frequent Model Predictions That Weren't Selected ===")
frequent_snubs = (
    snubs_df.groupby(name_col)
    .agg(times_snubbed=('season', 'count'), avg_prob=('rf_proba', 'mean'))
    .sort_values('times_snubbed', ascending=False)
    .head(10)
)
print(frequent_snubs.round(3).to_string())

# ── 14. PLAYER CAREER TRAJECTORY plots ───────────────────────────────────────
print("\n=== Player Career Probability Trajectories ===")

# Generate probabilities for ALL seasons (not just test), using RF
all_proba = best_rf.predict_proba(merged[FEATURE_COLS].fillna(0))[:, 1]
merged_plot = merged.copy()
merged_plot['rf_proba'] = all_proba
name_col_m = 'player' if 'player' in merged_plot.columns else 'player_id'

# Pick players: top 10 by total All-Star selections in the dataset
top_players = (
    merged_plot[merged_plot['all_star'] == 1]
    .groupby(name_col_m)['all_star'].sum()
    .sort_values(ascending=False)
    .head(8).index.tolist()
)

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for ax, player_name in zip(axes, top_players):
    player_data = (
        merged_plot[merged_plot[name_col_m] == player_name]
        .sort_values('label_season')
    )
    ax.plot(player_data['label_season'], player_data['rf_proba'],
            color='steelblue', linewidth=2, marker='o', markersize=4, label='Predicted prob')
    ax.fill_between(player_data['label_season'], player_data['rf_proba'],
                    alpha=0.15, color='steelblue')

    # Mark actual All-Star seasons with gold stars
    actual = player_data[player_data['all_star'] == 1]
    ax.scatter(actual['label_season'], actual['rf_proba'],
               color='gold', edgecolors='black', s=80, zorder=5, label='Actual All-Star')

    ax.set_title(player_name, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(All-Star)')
    ax.set_xlabel('Season')
    ax.axhline(0.5, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7, loc='lower left')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

plt.suptitle('Career All-Star Probability Trajectories (RF Model)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('career_trajectories.png', dpi=120)
plt.close()
print(f"Saved: career_trajectories.png  (players: {', '.join(top_players)})")

# ── 15. Demo ──────────────────────────────────────────────────────────────────
def predict_allstars(year: int, model=best_rf, min_games: int = MIN_GAMES):
    candidates = merged[merged['label_season'] == year].copy()
    if candidates.empty:
        print(f"No data for {year}.")
        return None
    name_col = 'player' if 'player' in candidates.columns else 'player_id'
    candidates['probability'] = model.predict_proba(candidates[FEATURE_COLS].fillna(0))[:, 1]
    eligible = candidates[candidates['current_g'] >= min_games]
    parts = []
    for conf, slots in [('East', 12), ('West', 12)]:
        parts.append(eligible[eligible['conference'] == conf].nlargest(slots, 'probability'))
    result = (
        pd.concat(parts)
        .sort_values('probability', ascending=False)
        [[name_col, 'label_season', 'team', 'conference', 'probability',
          'current_g', 'allnba_share', 'mvp_share', 'all_star']]
        .rename(columns={'label_season': 'season', 'all_star': 'actual_allstar'})
        .reset_index(drop=True)
    )
    result.index += 1
    return result

print("\n=== Demo: Top-24 Predicted All-Stars for 2024 (RF, conf-aware) ===")
demo = predict_allstars(2024)
if demo is not None:
    print(demo.round(3).to_string())
    correct = demo['actual_allstar'].sum()
    print(f"\n{correct}/24 were actual All-Stars ({correct/24*100:.0f}% top-24 recall)")
    print(f"Phase3 best for 2024: 16/25 (64%) — improvement: {correct - 16:+d}")
