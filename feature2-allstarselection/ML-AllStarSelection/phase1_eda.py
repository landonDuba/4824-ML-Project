"""
Phase 1 - Exploratory Data Analysis: All-Star Prediction
Goal: understand data shape, class imbalance, feature correlations,
and run a baseline model using previous-season stats to predict All-Star selection.
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ── 1. Load data ──────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("sumitrodatta/nba-aba-baa-stats")

allstar  = pd.read_csv(os.path.join(path, "All-Star Selections.csv"))
per_game = pd.read_csv(os.path.join(path, "Player Per Game.csv"))
advanced = pd.read_csv(os.path.join(path, "Advanced.csv"))

print("=== Dataset shapes ===")
print(f"All-Star Selections : {allstar.shape}")
print(f"Player Per Game     : {per_game.shape}")
print(f"Advanced            : {advanced.shape}")

# ── 2. Inspect All-Star data ──────────────────────────────────────────────────
print("\n=== All-Star Selections sample ===")
print(allstar.head(10).to_string())
print(f"\nLeagues present    : {allstar['lg'].unique()}")
print(f"Season range       : {allstar['season'].min()} – {allstar['season'].max()}")
print(f"'replaced' counts  :\n{allstar['replaced'].value_counts()}")

# Filter to NBA only (drop ABA, BAA eras for consistency)
allstar_nba = allstar[allstar['lg'] == 'NBA'].copy()
print(f"\nNBA All-Star rows   : {len(allstar_nba)}")
print(f"Selections per season (avg): {allstar_nba.groupby('season').size().mean():.1f}")

# ── 3. Build label column: was player an All-Star in season Y? ─────────────────
# We only care that the player was selected (including replacements counts).
allstar_flags = (
    allstar_nba[['player_id', 'season']]
    .drop_duplicates()
    .assign(all_star=1)
)

# ── 4. Merge per-game + advanced stats ───────────────────────────────────────
# For players traded mid-season, Basketball-Reference includes a 'TOT' (total) row.
# Keep only the TOT row when duplicates exist, otherwise keep the single entry.
def deduplicate_players(df):
    """Keep TOT row for traded players, else the single team row."""
    multi = df[df.duplicated(subset=['player_id', 'season'], keep=False)]
    tot   = multi[multi['team'] == 'TOT']
    single = df[~df.duplicated(subset=['player_id', 'season'], keep=False)]
    return pd.concat([single, tot], ignore_index=True)

per_game_clean = deduplicate_players(per_game[per_game['lg'] == 'NBA'])
advanced_clean = deduplicate_players(advanced[advanced['lg'] == 'NBA'])

print(f"\nAfter dedup - Per Game: {per_game_clean.shape}, Advanced: {advanced_clean.shape}")

# Merge on player_id + season
stats = per_game_clean.merge(
    advanced_clean[['player_id', 'season', 'per', 'ts_percent', 'ws', 'ws_48',
                    'bpm', 'vorp', 'usg_percent', 'obpm', 'dbpm']],
    on=['player_id', 'season'],
    how='inner'
)

# ── 5. Create lag features: stats from season Y-1, label from season Y ────────
stats_lag = stats.copy()
stats_lag['label_season'] = stats_lag['season'] + 1  # this row's stats predict NEXT season

merged = stats_lag.merge(
    allstar_flags.rename(columns={'season': 'label_season'}),
    on=['player_id', 'label_season'],
    how='left'
)
merged['all_star'] = merged['all_star'].fillna(0).astype(int)

# Filter to seasons where we actually have All-Star data (post-1951)
valid_seasons = allstar_flags['season'].unique()
merged = merged[merged['label_season'].isin(valid_seasons)]

print(f"\n=== Merged dataset ===")
print(f"Total player-seasons (with lag): {len(merged)}")
print(f"All-Stars labeled:               {merged['all_star'].sum()}")
print(f"Non-All-Stars:                   {(merged['all_star'] == 0).sum()}")
print(f"Class imbalance ratio:           1:{(merged['all_star']==0).sum() // merged['all_star'].sum()}")
print(f"Season range (label_season):     {merged['label_season'].min()} – {merged['label_season'].max()}")

# ── 6. Feature correlation with All-Star label ────────────────────────────────
feature_cols = [
    'pts_per_game', 'ast_per_game', 'trb_per_game', 'stl_per_game', 'blk_per_game',
    'fg_percent', 'ft_percent', 'mp_per_game', 'g',
    'per', 'ts_percent', 'ws', 'ws_48', 'bpm', 'vorp', 'usg_percent'
]

corr = merged[feature_cols + ['all_star']].corr()['all_star'].drop('all_star').sort_values(ascending=False)
print("\n=== Feature correlation with All-Star label ===")
print(corr.to_string())

plt.figure(figsize=(8, 6))
corr.plot(kind='barh', color=['steelblue' if v >= 0 else 'salmon' for v in corr])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Correlation with All-Star Selection')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=120)
plt.close()
print("\nSaved: feature_correlation.png")

# ── 7. Class imbalance by season ──────────────────────────────────────────────
season_counts = merged.groupby('label_season')['all_star'].agg(['sum', 'count'])
season_counts['rate'] = season_counts['sum'] / season_counts['count']

plt.figure(figsize=(14, 4))
plt.plot(season_counts.index, season_counts['rate'], marker='o', linewidth=1, markersize=3)
plt.ylabel('All-Star Selection Rate')
plt.xlabel('Season')
plt.title('All-Star Selection Rate per Season (NBA)')
plt.tight_layout()
plt.savefig('allstar_rate_by_season.png', dpi=120)
plt.close()
print("Saved: allstar_rate_by_season.png")

# ── 8. Baseline Logistic Regression ──────────────────────────────────────────
# Chronological split: train on seasons before 2015, test on 2015+
train = merged[merged['label_season'] < 2015].copy()
test  = merged[merged['label_season'] >= 2015].copy()

X_train = train[feature_cols].fillna(0)
y_train = train['all_star']
X_test  = test[feature_cols].fillna(0)
y_test  = test['all_star']

print(f"\n=== Train/Test Split ===")
print(f"Train seasons: {train['label_season'].min()} – {train['label_season'].max()}  |  {len(train)} rows  |  {y_train.sum()} All-Stars")
print(f"Test seasons : {test['label_season'].min()} – {test['label_season'].max()}   |  {len(test)} rows   |  {y_test.sum()} All-Stars")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Baseline Logistic Regression Results ===")
print(classification_report(y_test, y_pred, target_names=['Non-All-Star', 'All-Star']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-All-Star', 'All-Star'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False)
plt.title('Baseline Logistic Regression - Confusion Matrix')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=120)
plt.close()
print("Saved: baseline_confusion_matrix.png")

# ── 9. Top-N prediction sanity check ─────────────────────────────────────────
# For each test season, check how many true All-Stars appear in top-24 predicted
test = test.copy()
test['proba'] = y_proba

print("\n=== Top-24 Precision per Test Season ===")
for season, grp in test.groupby('label_season'):
    top24 = grp.nlargest(24, 'proba')
    hits  = top24['all_star'].sum()
    total = grp['all_star'].sum()
    print(f"  {season}: {hits}/{total} true All-Stars in top 24  ({hits/total*100:.0f}% recall)")
