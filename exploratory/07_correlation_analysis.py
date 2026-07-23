import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import save_fig

ENGINEERED_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'engineered_features', 'engineered_features.csv')


def figure_4_9_correlation_heatmap():
    df = pd.read_csv(ENGINEERED_FEATURES_PATH)

    exclude_cols = ['home_team', 'away_team', 'date', 'ftr', 'ftr_encoded', 'season']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    corr_matrix = df[numeric_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > 0.7:
                high_corr_pairs.append((idx, col, corr_matrix.loc[idx, col]))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = high_corr_pairs[:20]

    selected_features = list(set([p[0] for p in top_pairs] + [p[1] for p in top_pairs]))
    if len(selected_features) > 15:
        selected_features = selected_features[:15]

    fig, ax = plt.subplots(figsize=(14, 12))
    corr_subset = df[selected_features].corr()
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))

    sns.heatmap(
        corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
        center=0, ax=ax, linewidths=0.5, linecolor='#334155',
        vmin=-1, vmax=1,
        annot_kws={'size': 8},
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    ax.set_title('Correlation Heatmap of Pre-match Engineered Features', fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=9)
    fig.tight_layout()
    save_fig(fig, 'fig_4_9_correlation_heatmap')


if __name__ == '__main__':
    figure_4_9_correlation_heatmap()
