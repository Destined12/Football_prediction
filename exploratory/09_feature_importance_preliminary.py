import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import save_fig

FEATURE_IMPORTANCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'feature_importance.csv')


def figure_4_10_shap_feature_importance():
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    top_n = min(50, len(importance_df))
    top_features = importance_df.head(top_n).sort_values('shap_importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(top_features['feature'], top_features['shap_importance'], color=colors, edgecolor='#0f172a')
    ax.set_title('SHAP Feature Importance (Top 50 Selected Features)', fontweight='bold')
    ax.set_xlabel('SHAP Importance')
    ax.set_ylabel('Feature')

    for i, (val, name) in enumerate(zip(top_features['shap_importance'], top_features['feature'])):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8, color='#94a3b8')

    fig.tight_layout()
    save_fig(fig, 'fig_4_10_shap_feature_importance')


if __name__ == '__main__':
    figure_4_10_shap_feature_importance()
