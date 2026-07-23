import os
import sys
import time
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'web', 'trained_models')

META_COLS = ['home_team', 'away_team', 'date', 'ftr', 'ftr_encoded', 'season']
CORR_THRESHOLD = 0.90


def load_features():
    path = os.path.join(RESULTS_DIR, 'engineered_features', 'engineered_features.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df, feature_cols


def correlation_filter(df, feature_cols, threshold=CORR_THRESHOLD):
    print(f'\n--- Correlation Filter (|r| > {threshold}) ---')
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for corr_feat in correlated:
            if corr_feat not in to_drop and col not in to_drop:
                to_drop.add(corr_feat)

    kept = [c for c in feature_cols if c not in to_drop]
    print(f'  Dropped {len(to_drop)} highly correlated features')
    print(f'  Remaining: {len(kept)} features')
    if to_drop:
        print(f'  Dropped: {sorted(to_drop)[:20]}{"..." if len(to_drop) > 20 else ""}')
    return kept


def shap_importance_ranking(df, feature_cols, target='ftr_encoded'):
    print(f'\n--- SHAP Feature Importance Ranking ---')

    from catboost import CatBoostClassifier

    X = df[feature_cols].fillna(0).values
    y = df[target].values.astype(int)

    model = CatBoostClassifier(
        iterations=500, learning_rate=0.1, depth=5,
        random_state=42, verbose=0, loss_function='MultiClass',
        classes_count=3
    )
    model.fit(X, y)

    import shap
    shap_sample_size = min(200, len(X))
    X_sample = X[:shap_sample_size]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=1)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=1)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

    print('  Top 15 features by SHAP importance:')
    for rank, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f'    {rank:2d}. {row["feature"]}: {row["shap_importance"]:.4f}')

    return importance_df


def evaluate_feature_counts(df, ranked_features, target='ftr_encoded'):
    print(f'\n--- Evaluating Feature Counts via TimeSeriesSplit(5) ---')

    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, accuracy_score

    tscv = TimeSeriesSplit(n_splits=5)
    y = df[target].values.astype(int)

    candidate_counts = [30, 40, 50, 60]
    results = []

    for n_feats in candidate_counts:
        if n_feats > len(ranked_features):
            continue

        selected = ranked_features[:n_feats]
        X = df[selected].fillna(0).values

        fold_f1s = []
        fold_accs = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
                objective='multi:softprob', num_class=3,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            fold_f1s.append(f1_score(y_val, y_pred, average='macro'))
            fold_accs.append(accuracy_score(y_val, y_pred))

        mean_f1 = np.mean(fold_f1s)
        mean_acc = np.mean(fold_accs)
        results.append({
            'n_features': n_feats,
            'mean_macro_f1': mean_f1,
            'mean_accuracy': mean_acc,
            'std_macro_f1': np.std(fold_f1s),
        })
        print(f'  Top {n_feats:3d} features: CV macro F1={mean_f1:.4f} (+/-{np.std(fold_f1s):.4f}), accuracy={mean_acc:.4f}')

    best = max(results, key=lambda x: x['mean_macro_f1'])
    print(f'\n  Best: {best["n_features"]} features (macro F1={best["mean_macro_f1"]:.4f})')
    return best, results


def run_feature_selection():
    t_start = time.time()
    print('=' * 60)
    print('FEATURE SELECTION (v2)')
    print('=' * 60)

    df, feature_cols = load_features()
    print(f'Loaded {len(df)} rows, {len(feature_cols)} features')

    train_df = df[df['season'] != '2024-2025'].copy()
    print(f'Training data: {len(train_df)} rows (excluding test season 2024-2025)')

    # Step 1: Correlation filter (on training data only)
    filtered = correlation_filter(train_df, feature_cols)

    # Step 2: SHAP importance ranking (on training data only)
    importance_df = shap_importance_ranking(train_df, filtered)
    ranked_features = importance_df['feature'].tolist()

    # Step 3: Evaluate feature counts (full data with TimeSeriesSplit — CV handles leakage)
    best, all_results = evaluate_feature_counts(df, ranked_features)

    # Step 4: Save outputs
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    importance_df.to_csv(os.path.join(RESULTS_DIR, 'tables', 'feature_importance.csv'), index=False)

    selected_features = ranked_features[:best['n_features']]
    selected_df = pd.DataFrame({
        'rank': range(1, len(selected_features) + 1),
        'feature': selected_features,
    })
    selected_df.to_csv(os.path.join(RESULTS_DIR, 'tables', 'selected_features.csv'), index=False)

    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'selected_features.pkl'))

    eval_df = pd.DataFrame(all_results)
    eval_df.to_csv(os.path.join(RESULTS_DIR, 'tables', 'feature_selection_eval.csv'), index=False)

    print(f'\n  Saved: selected_features.csv ({len(selected_features)} features)')
    print(f'  Saved: feature_importance.csv')
    print(f'  Saved: feature_selection_eval.csv')
    print(f'\n  Total time: {time.time()-t_start:.1f}s')
    print('DONE')

    return df, selected_features


if __name__ == '__main__':
    run_feature_selection()
