import os
import sys
import time
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'web', 'trained_models')

META_COLS = ['home_team', 'away_team', 'date', 'ftr', 'ftr_encoded', 'season']


def load_data():
    path = os.path.join(RESULTS_DIR, 'engineered_features', 'engineered_features.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    selected_path = os.path.join(MODELS_DIR, 'selected_features.pkl')
    selected_features = joblib.load(selected_path)
    return df, selected_features


def chronological_split(df, test_season='2024-2025'):
    train = df[df['season'] != test_season].copy()
    test = df[df['season'] == test_season].copy()
    return train, test


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

    param_dist = {
        'n_estimators': [300, 500, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
    }

    print('  Tuning XGBoost (TimeSeriesSplit(5), n_jobs=1)...', flush=True)
    xgb = XGBClassifier(
        random_state=42, verbosity=0,
        objective='multi:softprob', num_class=3, eval_metric='mlogloss'
    )
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=10, cv=tscv,
        scoring='f1_macro', random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    print(f'    Best params: {search.best_params_}', flush=True)
    print(f'    Best CV macro F1: {search.best_score_:.4f}', flush=True)
    return search.best_estimator_


def train_svm(X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV

    param_dist = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto'],
        'svm__kernel': ['rbf'],
    }

    print('  Tuning SVM with Pipeline(Scaler, SVC) (TimeSeriesSplit(5), n_jobs=1)...', flush=True)
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=False, class_weight='balanced', random_state=42))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        svm_pipeline, param_dist, n_iter=6, cv=tscv,
        scoring='f1_macro', random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    print(f'    Best params: {search.best_params_}', flush=True)
    print(f'    Best CV macro F1: {search.best_score_:.4f}', flush=True)

    print('  Calibrating SVM probabilities (CalibratedClassifierCV)...', flush=True)
    calibrated = CalibratedClassifierCV(
        search.best_estimator_, method='isotonic', cv=3
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

    param_dist = {
        'n_estimators': [300, 500, 700],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    print('  Tuning Random Forest (TimeSeriesSplit(5), n_jobs=1)...', flush=True)
    rf = RandomForestClassifier(
        class_weight='balanced', random_state=42, n_jobs=1
    )
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=tscv,
        scoring='f1_macro', random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    print(f'    Best params: {search.best_params_}', flush=True)
    print(f'    Best CV macro F1: {search.best_score_:.4f}', flush=True)
    return search.best_estimator_


def train_meta_learner(models, X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score
    from sklearn.base import clone as sklearn_clone

    tscv = TimeSeriesSplit(n_splits=5)

    print('  Generating out-of-fold stacking features...', flush=True)
    oof_predictions = np.zeros((len(X_train), len(models) * 3))

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f'    Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}', flush=True)
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        fold_probas = []
        for model in models:
            m = sklearn_clone(model)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_val)
            fold_probas.append(proba)
        oof_predictions[val_idx] = np.concatenate(fold_probas, axis=1)

    print(f'  OOF stacking features shape: {oof_predictions.shape}', flush=True)

    candidates = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            solver='lbfgs', max_iter=5000, C=1.0, random_state=42
        ),
        'LogisticRegression_C01': LogisticRegression(
            class_weight='balanced',
            solver='lbfgs', max_iter=5000, C=0.1, random_state=42
        ),
        'CalibratedLinearSVM': CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
            method='isotonic', cv=3
        ),
    }

    best_name = None
    best_score = -1
    best_model = None

    print('\n  Evaluating meta-learner candidates (on OOF predictions):', flush=True)
    for name, model in candidates.items():
        fold_scores = []
        for train_idx, val_idx in tscv.split(oof_predictions):
            X_tr, X_val = oof_predictions[train_idx], oof_predictions[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            m = sklearn_clone(model)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='macro'))

        mean_score = np.mean(fold_scores)
        print(f'    {name}: CV macro F1={mean_score:.4f} (+/-{np.std(fold_scores):.4f})', flush=True)

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = m

    print(f'\n  Best meta-learner: {best_name} (macro F1={best_score:.4f})', flush=True)

    print('  Retraining base models on full training data for final inference...', flush=True)
    final_probas = []
    for model in models:
        m = sklearn_clone(model)
        m.fit(X_train, y_train)
        final_probas.append(m.predict_proba(X_train))
    final_stack = np.concatenate(final_probas, axis=1)

    best_model.fit(final_stack, y_train)

    print('  Retraining base models and saving them for inference...', flush=True)
    for model, name in zip(models, ['xgboost', 'svm', 'random_forest']):
        m = sklearn_clone(model)
        m.fit(X_train, y_train)
        key = 'xgboost_model' if name == 'xgboost' else f'{name}_model' if name == 'random_forest' else 'svm_pipeline'
        joblib.dump(m, os.path.join(MODELS_DIR, f'{key}.pkl'), compress=3)
        print(f'    Retrained and saved {name}', flush=True)

    return best_model


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'

    df, selected_features = load_data()
    train_df, test_df = chronological_split(df)
    X_train = train_df[selected_features].fillna(0).values
    y_train = train_df['ftr_encoded'].values.astype(int)
    X_test = test_df[selected_features].fillna(0).values
    y_test = test_df['ftr_encoded'].values.astype(int)
    print(f'Data loaded: {len(train_df)} train, {len(test_df)} test, {len(selected_features)} features', flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if step in ('xgb', 'all'):
        print('\n=== TRAINING XGBoost ===', flush=True)
        t0 = time.time()
        xgb_model = train_xgboost(X_train, y_train)
        joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'), compress=3)
        print(f'  XGBoost saved ({time.time()-t0:.1f}s)', flush=True)

    if step in ('svm', 'all'):
        print('\n=== TRAINING SVM ===', flush=True)
        t0 = time.time()
        svm_model = train_svm(X_train, y_train)
        joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_pipeline.pkl'), compress=3)
        print(f'  SVM saved ({time.time()-t0:.1f}s)', flush=True)

    if step in ('rf', 'all'):
        print('\n=== TRAINING Random Forest ===', flush=True)
        t0 = time.time()
        rf_model = train_random_forest(X_train, y_train)
        joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.pkl'), compress=3)
        print(f'  Random Forest saved ({time.time()-t0:.1f}s)', flush=True)

    if step in ('meta', 'all'):
        print('\n=== TRAINING META-LEARNER ===', flush=True)
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_pipeline.pkl'))
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))

        t0 = time.time()
        meta_model = train_meta_learner([xgb_model, svm_model, rf_model], X_train, y_train)
        joblib.dump(meta_model, os.path.join(MODELS_DIR, 'logistic_meta_model.pkl'), compress=3)
        print(f'  Meta-learner saved ({time.time()-t0:.1f}s)', flush=True)

    print('\nDone!', flush=True)
