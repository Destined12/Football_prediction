import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'web', 'trained_models')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'explainability')


def run_explainability(base_models, X_test, selected_features):
    print('=' * 60)
    print('MODEL EXPLAINABILITY (SHAP)')
    print('=' * 60)

    import shap

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('\n--- SHAP Summary Plot ---')
    cat_model = base_models['catboost']
    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=1)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=1)

    importance_df = pd.DataFrame({
        'feature': selected_features,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False)

    print('  Top 10 SHAP features:')
    for _, row in importance_df.head(10).iterrows():
        print(f'    {row["feature"]}: {row["shap_importance"]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = importance_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_n)))
    ax.barh(range(len(top_n)), top_n['shap_importance'].values[::-1], color=colors, edgecolor='#0f172a')
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['feature'].values[::-1])
    ax.set_title('SHAP Feature Importance (CatBoost)', fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'shap_feature_importance.svg'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved shap_feature_importance.png/svg')

    print('\n--- SHAP Beeswarm Plot ---')
    try:
        shap_df = pd.DataFrame(X_test, columns=selected_features)
        shap_exp = shap.Explanation(
            values=shap_values if isinstance(shap_values, np.ndarray) else shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0],
            data=shap_df.values,
            feature_names=selected_features
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        shap.plots.beeswarm(shap_exp, max_display=20, show=False)
        plt.title('SHAP Summary Plot', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_beeswarm.svg'), bbox_inches='tight')
        plt.close('all')
        print('  Saved shap_summary_beeswarm.png/svg')
    except Exception as e:
        print(f'  Beeswarm plot skipped: {e}')

    print('\n--- SHAP Waterfall Plot (Sample Prediction) ---')
    try:
        sample_idx = 0
        if isinstance(shap_values, list):
            sample_shap = shap_values[0][sample_idx]
        else:
            sample_shap = shap_values[sample_idx]

        sample_exp = shap.Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test[sample_idx],
            feature_names=selected_features
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(sample_exp, max_display=15, show=False)
        plt.title('SHAP Waterfall Plot (Single Prediction)', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.svg'), bbox_inches='tight')
        plt.close('all')
        print('  Saved shap_waterfall.png/svg')
    except Exception as e:
        print(f'  Waterfall plot skipped: {e}')

    joblib.dump(importance_df, os.path.join(MODELS_DIR, 'shap_feature_importance.pkl'), compress=3)
    print('\n  All SHAP plots and data saved.')


if __name__ == '__main__':
    print("Run via model/run_all.py instead.")
