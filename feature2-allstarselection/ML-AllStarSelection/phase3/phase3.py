"""Phase 3 - Enhanced features, GBM, conference-aware top-24."""

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
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve,
)
import shap

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

win_pct = team_summary[team_summary['lg'] == 'NBA'][['season', 'abbreviation', 'w', 'l']].copy()
win_pct['win_pct'] = win_pct['w'] / (win_pct['w'] + win_pct['l'])
win_pct = win_pct.rename(columns={'abbreviation': 'team'})

stats = stats.merge(win_pct[['season', 'team', 'win_pct']], on=['season', 'team'], how='left')
stats['win_pct'] = stats['win_pct'].fillna(0.5)  # TOT rows have no team — use league average

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

for lag in [1, 2, 3]:
    temp = allstar_flags[['player_id', 'season']].copy()
    temp['label_season'] = temp['season'] + lag
    temp[f'was_as_{lag}'] = 1
    merged = merged.merge(
        temp[['player_id', 'label_season', f'was_as_{lag}']],
        on=['player_id', 'label_season'],
        how='left'
    )
    merged[f'was_as_{lag}'] = merged[f'was_as_{lag}'].fillna(0).astype(int)

merged['prior_as_3yr'] = merged['was_as_1'] + merged['was_as_2'] + merged['was_as_3']

EAST = {
    'ATL','BOS','BRK','CHI','CHO','CLE','DET','IND','MIA','MIL',
    'NJN','NYK','ORL','PHI','TOR','WAS','CHH','CHA'
}
WEST = {
    'DAL','DEN','GSW','HOU','LAC','LAL','MEM','MIN','NOP','OKC',
    'PHO','POR','SAC','SAS','UTA','SEA','NOH','NOK','VAN'
}
merged['conference'] = merged['team'].apply(
    lambda t: 'East' if t in EAST else ('West' if t in WEST else 'Unknown')
)

BASE_COLS = [
    'pts_per_game', 'ast_per_game', 'trb_per_game', 'stl_per_game', 'blk_per_game',
    'fg_percent', 'ft_percent', 'mp_per_game', 'g',
    'per', 'ts_percent', 'ws', 'ws_48', 'bpm', 'vorp', 'usg_percent'
]
NEW_COLS = (
    [f'{c}_2yr'   for c in roll_cols] +
    [f'{c}_delta' for c in delta_cols] +
    ['win_pct', 'prior_as_3yr', 'was_as_1', 'was_as_2', 'was_as_3']
)
FEATURE_COLS = BASE_COLS + NEW_COLS

print(f"Total features: {len(FEATURE_COLS)}")

train = merged[merged['label_season'] < 2015].copy()
test  = merged[merged['label_season'] >= 2015].copy()

X_train = train[FEATURE_COLS].fillna(0)
y_train = train['all_star']
X_test  = test[FEATURE_COLS].fillna(0)
y_test  = test['all_star']

print(f"Train: {len(train)} rows | {y_train.sum()} All-Stars")
print(f"Test : {len(test)} rows  | {y_test.sum()} All-Stars\n")

tscv = TimeSeriesSplit(n_splits=5)

print("=== Tuning Logistic Regression ===")
lr_cv = GridSearchCV(
    Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    {'clf__C': [0.01, 0.1, 1.0, 10.0]},
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
lr_cv.fit(X_train, y_train)
best_lr   = lr_cv.best_estimator_
lr_proba  = best_lr.predict_proba(X_test)[:, 1]
lr_pred   = best_lr.predict(X_test)
print(f"Best C: {lr_cv.best_params_['clf__C']}  |  CV AUC: {lr_cv.best_score_:.4f}")

print("\n=== Tuning Random Forest ===")
rf_cv = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {
        'n_estimators':     [200, 400],
        'max_depth':        [6, 10, None],
        'min_samples_leaf': [1, 5, 10],
    },
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
rf_cv.fit(X_train, y_train)
best_rf  = rf_cv.best_estimator_
rf_proba = best_rf.predict_proba(X_test)[:, 1]
rf_pred  = best_rf.predict(X_test)
print(f"Best params: {rf_cv.best_params_}  |  CV AUC: {rf_cv.best_score_:.4f}")

print("\n=== Tuning Gradient Boosting ===")
# HistGradientBoosting doesn't support class_weight — use sample_weight instead.
sample_weight = y_train.map({0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()})

hgb_cv = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    {
        'max_iter':      [300, 500],
        'max_depth':     [4, 6],
        'learning_rate': [0.05, 0.1],
    },
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
hgb_cv.fit(X_train, y_train, sample_weight=sample_weight)
best_xgb  = hgb_cv.best_estimator_
xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
xgb_pred  = best_xgb.predict(X_test)
print(f"Best params: {hgb_cv.best_params_}  |  CV AUC: {hgb_cv.best_score_:.4f}")

print("\n=== Head-to-Head Evaluation (standard metrics) ===")

def evaluate(name, y_true, y_pred, y_proba):
    return {
        'Model':     name,
        'ROC-AUC':   round(roc_auc_score(y_true, y_proba), 4),
        'F1':        round(f1_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
    }

results = pd.DataFrame([
    evaluate('LR',               y_test, lr_pred,  lr_proba),
    evaluate('RF',               y_test, rf_pred,  rf_proba),
    evaluate('GradientBoosting', y_test, xgb_pred, xgb_proba),
])
print(results.to_string(index=False))

print("\n=== Conference-Aware Top-24 Recall (12 East + 12 West) ===")

test_eval = test.copy()
test_eval['lr_proba']  = lr_proba
test_eval['rf_proba']  = rf_proba
test_eval['xgb_proba'] = xgb_proba

season_rows = []
for season, grp in test_eval.groupby('label_season'):
    total = grp['all_star'].sum()
    lr_hits = xgb_hits = rf_hits = 0
    for conf, slots in [('East', 12), ('West', 12)]:
        cg = grp[grp['conference'] == conf]
        if cg.empty:
            continue
        lr_hits  += cg.nlargest(slots, 'lr_proba')['all_star'].sum()
        rf_hits  += cg.nlargest(slots, 'rf_proba')['all_star'].sum()
        xgb_hits += cg.nlargest(slots, 'xgb_proba')['all_star'].sum()
    season_rows.append({
        'season': season, 'true_AS': total,
        'LR': lr_hits, 'RF': rf_hits, 'XGB': xgb_hits
    })

season_df = pd.DataFrame(season_rows)
print(f"\n{'Season':>8}  {'True AS':>7}  {'LR':>6}  {'RF':>6}  {'XGB':>6}")
for _, r in season_df.iterrows():
    t = r['true_AS']
    print(f"  {int(r['season'])}: "
          f"LR {int(r['LR'])}/{int(t)} ({int(r['LR']/t*100)}%)  |  "
          f"RF {int(r['RF'])}/{int(t)} ({int(r['RF']/t*100)}%)  |  "
          f"XGB {int(r['XGB'])}/{int(t)} ({int(r['XGB']/t*100)}%)")

avg_lr  = season_df['LR'].sum()  / season_df['true_AS'].sum()
avg_rf  = season_df['RF'].sum()  / season_df['true_AS'].sum()
avg_xgb = season_df['XGB'].sum() / season_df['true_AS'].sum()
print(f"\nOverall top-24 recall (conf-aware):  LR {avg_lr:.1%}  |  RF {avg_rf:.1%}  |  XGB {avg_xgb:.1%}")

x = np.arange(len(season_df))
w = 0.25
plt.figure(figsize=(13, 5))
plt.bar(x - w,   season_df['LR'],  w, label='LR',     color='steelblue')
plt.bar(x,       season_df['RF'],  w, label='RF',     color='darkorange')
plt.bar(x + w,   season_df['XGB'], w, label='XGBoost',color='seagreen')
plt.xticks(x, season_df['season'].astype(int), rotation=45)
plt.ylabel('True All-Stars in Top 24 (conf-aware)')
plt.title('Phase 3 — Conference-Aware Top-24 Recall per Season')
plt.legend()
plt.tight_layout()
plt.savefig('phase3_top24_recall.png', dpi=120)
plt.close()
print("\nSaved: phase3_top24_recall.png")

plt.figure(figsize=(6, 5))
for name, proba, color in [
    ('LR',               lr_proba,  'steelblue'),
    ('RF',               rf_proba,  'darkorange'),
    ('GradientBoosting', xgb_proba, 'seagreen'),
]:
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--',linewidth=1,label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Phase 3 — ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('phase3_roc.png', dpi=120)
plt.close()
print("Saved: phase3_roc.png")

print("\n=== SHAP Feature Attribution (Random Forest) ===")
explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
print("\nTop 10 most influential features (RF via SHAP):")
print(mean_shap.head(10).round(4).to_string())

plt.figure(figsize=(8, 6))
mean_shap.head(15).plot(kind='barh', color='seagreen')
plt.gca().invert_yaxis()
plt.xlabel('Mean |SHAP value|')
plt.title('Top 15 Feature Importance — Phase 3 Enhanced Features (RF)')
plt.tight_layout()
plt.savefig('phase3_shap.png', dpi=120)
plt.close()
print("Saved: phase3_shap.png")

def predict_allstars(year: int, model=best_xgb, top_n: int = 24, conf_aware: bool = True):
    candidates = merged[merged['label_season'] == year].copy()
    if candidates.empty:
        print(f"No data for {year}. Range: {merged['label_season'].min()}–{merged['label_season'].max()}")
        return None

    name_col = 'player' if 'player' in candidates.columns else 'player_id'
    candidates['probability'] = model.predict_proba(candidates[FEATURE_COLS].fillna(0))[:, 1]

    if conf_aware:
        parts = []
        for conf, slots in [('East', 12), ('West', 12)]:
            cg = candidates[candidates['conference'] == conf]
            parts.append(cg.nlargest(slots, 'probability'))
        result = pd.concat(parts).sort_values('probability', ascending=False)
    else:
        result = candidates.nlargest(top_n, 'probability')

    result = (
        result[[name_col, 'label_season', 'team', 'conference', 'probability', 'all_star']]
        .rename(columns={'label_season': 'season', 'all_star': 'actual_allstar'})
        .reset_index(drop=True)
    )
    result.index += 1
    return result

print("\n=== Demo: Top-24 Predicted All-Stars for 2024 (GradientBoosting, conference-aware) ===")
demo = predict_allstars(2024)
if demo is not None:
    print(demo.to_string())
    correct = demo['actual_allstar'].sum()
    print(f"\n{correct}/24 were actual All-Stars ({correct/24*100:.0f}% top-24 recall)")
