"""
Phase 2 - Final Implementation: All-Star Prediction
Tuned LR vs Random Forest, SHAP attribution, ethics audit,
and live demo: input year Y -> top-24 predicted All-Stars.
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, f1_score,
    precision_score, recall_score
)
import shap

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
path = kagglehub.dataset_download("sumitrodatta/nba-aba-baa-stats")

allstar  = pd.read_csv(os.path.join(path, "All-Star Selections.csv"))
per_game = pd.read_csv(os.path.join(path, "Player Per Game.csv"))
advanced = pd.read_csv(os.path.join(path, "Advanced.csv"))

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

FEATURE_COLS = [
    'pts_per_game', 'ast_per_game', 'trb_per_game', 'stl_per_game', 'blk_per_game',
    'fg_percent', 'ft_percent', 'mp_per_game', 'g',
    'per', 'ts_percent', 'ws', 'ws_48', 'bpm', 'vorp', 'usg_percent'
]

# ── 2. Train / test split (chronological) ─────────────────────────────────────
train = merged[merged['label_season'] < 2015].copy()
test  = merged[merged['label_season'] >= 2015].copy()

X_train = train[FEATURE_COLS].fillna(0)
y_train = train['all_star']
X_test  = test[FEATURE_COLS].fillna(0)
y_test  = test['all_star']

print(f"Train: {len(train)} rows | {y_train.sum()} All-Stars")
print(f"Test : {len(test)} rows  | {y_test.sum()} All-Stars\n")

# ── 3. Tuned Logistic Regression ──────────────────────────────────────────────
print("=== Tuning Logistic Regression ===")
tscv = TimeSeriesSplit(n_splits=5)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
lr_cv = GridSearchCV(
    lr_pipeline,
    {'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]},
    cv=tscv, scoring='roc_auc', n_jobs=-1
)
lr_cv.fit(X_train, y_train)
best_lr = lr_cv.best_estimator_
print(f"Best C: {lr_cv.best_params_['clf__C']}  |  CV AUC: {lr_cv.best_score_:.4f}")

lr_pred  = best_lr.predict(X_test)
lr_proba = best_lr.predict_proba(X_test)[:, 1]

# ── 4. Tuned Random Forest ────────────────────────────────────────────────────
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
best_rf = rf_cv.best_estimator_
print(f"Best params: {rf_cv.best_params_}  |  CV AUC: {rf_cv.best_score_:.4f}")

rf_pred  = best_rf.predict(X_test)
rf_proba = best_rf.predict_proba(X_test)[:, 1]

# ── 5. Head-to-head evaluation table ─────────────────────────────────────────
print("\n=== Head-to-Head Evaluation ===")

def evaluate(name, y_true, y_pred, y_proba):
    return {
        'Model':     name,
        'ROC-AUC':   round(roc_auc_score(y_true, y_proba), 4),
        'F1':        round(f1_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
    }

results = pd.DataFrame([
    evaluate('Logistic Regression', y_test, lr_pred, lr_proba),
    evaluate('Random Forest',       y_test, rf_pred, rf_proba),
])
print(results.to_string(index=False))

print("\nLogistic Regression — full report:")
print(classification_report(y_test, lr_pred, target_names=['Non-All-Star', 'All-Star']))
print("Random Forest — full report:")
print(classification_report(y_test, rf_pred, target_names=['Non-All-Star', 'All-Star']))

# ── 6. ROC curve comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, proba, color) in zip(axes, [
    ('Logistic Regression', lr_proba, 'steelblue'),
    ('Random Forest',       rf_proba, 'darkorange'),
]):
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=120)
plt.close()
print("\nSaved: roc_comparison.png")

# ── 7. Confusion matrices side by side ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, pred, color) in zip(axes, [
    ('Logistic Regression', lr_pred, 'Blues'),
    ('Random Forest',       rf_pred, 'Oranges'),
]):
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=['Non-AS', 'All-Star']).plot(ax=ax, colorbar=False, cmap=color)
    ax.set_title(f'{name}\nF1 (All-Star): {f1:.3f}')
plt.tight_layout()
plt.savefig('confusion_comparison.png', dpi=120)
plt.close()
print("Saved: confusion_comparison.png")

# ── 8. Top-24 recall per season ───────────────────────────────────────────────
print("\n=== Top-24 Recall per Season ===")
test_eval = test.copy()
test_eval['lr_proba'] = lr_proba
test_eval['rf_proba'] = rf_proba

season_rows = []
for season, grp in test_eval.groupby('label_season'):
    total   = grp['all_star'].sum()
    lr_hits = grp.nlargest(24, 'lr_proba')['all_star'].sum()
    rf_hits = grp.nlargest(24, 'rf_proba')['all_star'].sum()
    season_rows.append({'season': season, 'true_AS': total, 'LR': lr_hits, 'RF': rf_hits})
    print(f"  {season}: LR {lr_hits}/{total} ({lr_hits/total*100:.0f}%)  |  RF {rf_hits}/{total} ({rf_hits/total*100:.0f}%)")

season_df = pd.DataFrame(season_rows)
x = np.arange(len(season_df))
w = 0.35
plt.figure(figsize=(11, 4))
plt.bar(x - w/2, season_df['LR'], w, label='Logistic Regression', color='steelblue')
plt.bar(x + w/2, season_df['RF'], w, label='Random Forest',       color='darkorange')
plt.xticks(x, season_df['season'], rotation=45)
plt.ylabel('True All-Stars in Top 24 Predictions')
plt.title('Top-24 Recall per Season')
plt.legend()
plt.tight_layout()
plt.savefig('top24_recall.png', dpi=120)
plt.close()
print("Saved: top24_recall.png")

# ── 9. SHAP feature attribution (Random Forest) ───────────────────────────────
print("\n=== SHAP Feature Attribution ===")
explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# shap returns list [class0, class1], 3D array (n_samples, n_features, n_classes), or 2D array
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(sv, X_test, feature_names=FEATURE_COLS, show=False, plot_size=(9, 6))
plt.title('SHAP Feature Impact — Random Forest (All-Star class)')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
print("\nMean |SHAP| per feature (higher = more influential):")
print(mean_shap.round(4).to_string())

# Bar chart version of SHAP importance
plt.figure(figsize=(8, 5))
mean_shap.plot(kind='barh', color='darkorange')
plt.gca().invert_yaxis()
plt.xlabel('Mean |SHAP value|')
plt.title('Feature Importance via SHAP — Random Forest')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=120)
plt.close()
print("Saved: shap_importance.png")

# ── 10. Ethics audit ──────────────────────────────────────────────────────────
print("\n=== Ethics Audit ===")

EAST = {
    'ATL','BOS','BRK','CHI','CHO','CLE','DET','IND','MIA','MIL',
    'NJN','NYK','ORL','PHI','TOR','WAS','CHH','CHA'
}
WEST = {
    'DAL','DEN','GSW','HOU','LAC','LAL','MEM','MIN','NOP','OKC',
    'PHO','POR','SAC','SAS','UTA','SEA','NOH','NOK','VAN'
}
LARGE_MARKET = {'LAL','LAC','GSW','NYK','BOS','CHI','MIA','BRK','HOU','PHI'}

test_audit = test_eval.copy()
test_audit['conference'] = test_audit['team'].apply(
    lambda t: 'East' if t in EAST else ('West' if t in WEST else 'Unknown')
)
test_audit['market'] = test_audit['team'].apply(
    lambda t: 'Large' if t in LARGE_MARKET else 'Small/Mid'
)
test_audit['rf_top24'] = False
for season, grp in test_audit.groupby('label_season'):
    top_idx = grp.nlargest(24, 'rf_proba').index
    test_audit.loc[top_idx, 'rf_top24'] = True

def bias_table(df, group_col):
    return df.groupby(group_col).apply(
        lambda g: pd.Series({
            'n_players':         len(g),
            'true_AS_rate':      round(g['all_star'].mean(), 4),
            'predicted_AS_rate': round(g['rf_top24'].mean(), 4),
            'delta':             round(g['rf_top24'].mean() - g['all_star'].mean(), 4),
        })
    )

if 'pos' in test_audit.columns:
    print("\nPosition bias (positive delta = model over-selects this group):")
    print(bias_table(test_audit, 'pos').to_string())

print("\nConference bias:")
print(bias_table(test_audit, 'conference').to_string())

print("\nMarket size bias:")
print(bias_table(test_audit, 'market').to_string())

# ── 11. Demo: predict All-Stars for any season ────────────────────────────────
def predict_allstars(year: int, model=best_rf, top_n: int = 24):
    """Input: season year. Output: top_n ranked predicted All-Stars with probabilities."""
    candidates = merged[merged['label_season'] == year].copy()
    if candidates.empty:
        print(f"No data for season {year}. Available range: "
              f"{merged['label_season'].min()}–{merged['label_season'].max()}")
        return None

    name_col = 'player' if 'player' in candidates.columns else 'player_id'
    X = candidates[FEATURE_COLS].fillna(0)
    candidates = candidates.copy()
    candidates['probability'] = model.predict_proba(X)[:, 1]

    result = (
        candidates
        .nlargest(top_n, 'probability')
        [[name_col, 'label_season', 'team', 'probability', 'all_star']]
        .rename(columns={'label_season': 'season', 'all_star': 'actual_allstar'})
        .reset_index(drop=True)
    )
    result.index += 1
    return result

print("\n=== Demo: Top-24 Predicted All-Stars for 2024 ===")
demo = predict_allstars(2024)
if demo is not None:
    print(demo.to_string())
    correct = demo['actual_allstar'].sum()
    print(f"\n{correct}/24 were actual All-Stars ({correct/24*100:.0f}% top-24 recall)")
