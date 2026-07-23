import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'web', 'trained_models')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'explainability')


def load_data():
    path = os.path.join(RESULTS_DIR, 'engineered_features', 'engineered_features.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    selected_path = os.path.join(MODELS_DIR, 'selected_features.pkl')
    selected_features = joblib.load(selected_path)
    return df, selected_features


def chronological_split(df, test_season='2024-2025'):
    test = df[df['season'] == test_season].copy()
    return test


def run_shap_explainability():
    t_start = time.time()
    print('=' * 60, flush=True)
    print('MODEL EXPLAINABILITY (SHAP) — XGBoost', flush=True)
    print('=' * 60, flush=True)

    import shap

    df, selected_features = load_data()
    test_df = chronological_split(df)
    X_test = test_df[selected_features].fillna(0).values
    print(f'Test set: {len(test_df)} matches, {len(selected_features)} features', flush=True)

    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    print('XGBoost model loaded.', flush=True)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('\n--- Computing SHAP Values ---', flush=True)
    shap_sample_size = min(50, len(X_test))
    shap_sample = X_test[:shap_sample_size]
    print(f'  Using {shap_sample_size} samples for SHAP', flush=True)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(shap_sample)

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

    print('  Top 10 SHAP features:', flush=True)
    for _, row in importance_df.head(10).iterrows():
        print(f'    {row["feature"]}: {row["shap_importance"]:.4f}', flush=True)

    print('\n--- SHAP Feature Importance Bar Plot ---', flush=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = importance_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_n)))
    ax.barh(range(len(top_n)), top_n['shap_importance'].values[::-1], color=colors, edgecolor='#0f172a')
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['feature'].values[::-1])
    ax.set_title('SHAP Feature Importance (XGBoost)', fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'shap_feature_importance.svg'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved shap_feature_importance.png/svg', flush=True)

    print('\n--- SHAP Beeswarm Plot ---', flush=True)
    try:
        shap_df = pd.DataFrame(shap_sample, columns=selected_features)
        if isinstance(shap_values, list):
            sv_for_beeswarm = shap_values[0]
        else:
            sv_for_beeswarm = shap_values

        if sv_for_beeswarm.ndim == 3:
            sv_for_beeswarm = sv_for_beeswarm.mean(axis=2)

        shap_exp = shap.Explanation(
            values=sv_for_beeswarm,
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=shap_df.values,
            feature_names=selected_features
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        shap.plots.beeswarm(shap_exp, max_display=20, show=False)
        plt.title('SHAP Summary Plot (XGBoost)', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_beeswarm.svg'), bbox_inches='tight')
        plt.close('all')
        print('  Saved shap_summary_beeswarm.png/svg', flush=True)
    except Exception as e:
        print(f'  Beeswarm plot skipped: {e}', flush=True)

    print('\n--- SHAP Waterfall Plot (Sample Prediction) ---', flush=True)
    try:
        sample_idx = 0
        if isinstance(shap_values, list):
            sample_shap = shap_values[0][sample_idx]
        else:
            sample_shap = shap_values[sample_idx]

        if sample_shap.ndim > 1:
            sample_shap = sample_shap.mean(axis=1)

        sample_exp = shap.Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=shap_sample[sample_idx],
            feature_names=selected_features
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(sample_exp, max_display=15, show=False)
        plt.title('SHAP Waterfall Plot (Single Prediction)', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.svg'), bbox_inches='tight')
        plt.close('all')
        print('  Saved shap_waterfall.png/svg', flush=True)
    except Exception as e:
        print(f'  Waterfall plot skipped: {e}', flush=True)

    joblib.dump(importance_df, os.path.join(MODELS_DIR, 'shap_feature_importance.pkl'), compress=3)

    print(f'\n  Total time: {time.time()-t_start:.1f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    run_shap_explainability()
