import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from scipy.stats import randint, uniform

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


def tune_and_train_catboost(X_train, y_train):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        'depth': randint(4, 9),
        'learning_rate': uniform(0.03, 0.22),
        'iterations': randint(400, 1200),
        'l2_leaf_reg': uniform(1, 4),
        'bagging_temperature': uniform(0, 1.5),
        'random_strength': uniform(0, 1.5),
    }

    print('  Tuning CatBoost with RandomizedSearchCV (5 iters, 2-fold)...')
    cat = CatBoostClassifier(
        random_state=42, verbose=0, loss_function='MultiClass',
        classes_count=3
    )

    search = RandomizedSearchCV(
        cat, param_dist, n_iter=5, cv=2,
        scoring='accuracy', random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)

    print(f'    Best params: {search.best_params_}')
    print(f'    Best CV accuracy: {search.best_score_:.4f}')
    return search.best_estimator_


def tune_and_train_lightgbm(X_train, y_train):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    configs = [
        {'num_leaves': 15, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_samples': 15, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'num_rounds': 400},
        {'num_leaves': 25, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.3, 'reg_lambda': 0.3, 'num_rounds': 500},
        {'num_leaves': 35, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_samples': 10, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.0, 'reg_lambda': 0.6, 'num_rounds': 600},
        {'num_leaves': 20, 'learning_rate': 0.15, 'max_depth': 3, 'min_child_samples': 25, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0.6, 'reg_lambda': 0.0, 'num_rounds': 300},
        {'num_leaves': 45, 'learning_rate': 0.08, 'max_depth': 9, 'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.2, 'reg_lambda': 0.5, 'num_rounds': 500},
    ]

    print('  Manual tuning LightGBM (5 configs, 2-fold CV)...')
    best_score = 0
    best_cfg = None
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for i, cfg in enumerate(configs):
        num_rounds = cfg.pop('num_rounds')
        params = {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'seed': 42, **cfg}
        fold_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            dtrain = lgb.Dataset(X_train[train_idx], label=y_train[train_idx])
            dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
            booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
            preds = booster.predict(X_train[val_idx])
            fold_scores.append(np.mean(np.argmax(preds, axis=1) == y_train[val_idx]))
        mean_score = np.mean(fold_scores)
        cfg['num_rounds'] = num_rounds
        print(f'    Config {i+1}: CV={mean_score:.4f}')
        if mean_score > best_score:
            best_score = mean_score
            best_cfg = cfg.copy()

    print(f'    Best CV accuracy: {best_score:.4f}')
    final_params = {k: v for k, v in best_cfg.items() if k != 'num_rounds'}
    final_params.update({'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'seed': 42})
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(final_params, dtrain_full, num_boost_round=best_cfg['num_rounds'])

    class LGBWrapper:
        def __init__(self, booster, params, num_rounds):
            self.booster = booster
            self.params = params
            self.num_rounds = num_rounds
            self.classes_ = np.array([0, 1, 2])
        def predict_proba(self, X):
            return self.booster.predict(X)
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    return LGBWrapper(booster, final_params, best_cfg['num_rounds'])


def tune_and_train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        'n_estimators': randint(400, 1200),
        'learning_rate': uniform(0.03, 0.22),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 8),
        'subsample': uniform(0.6, 0.35),
        'colsample_bytree': uniform(0.6, 0.35),
        'gamma': uniform(0, 0.8),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1.5),
    }

    print('  Tuning XGBoost with RandomizedSearchCV (5 iters, 2-fold)...')
    xgb = XGBClassifier(
        random_state=42, verbosity=0, objective='multi:softprob',
        num_class=3, eval_metric='mlogloss'
    )

    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=5, cv=2,
        scoring='accuracy', random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)

    print(f'    Best params: {search.best_params_}')
    print(f'    Best CV accuracy: {search.best_score_:.4f}')
    return search.best_estimator_


def get_stacking_features(models, X):
    probas = []
    for name in ['catboost', 'lightgbm', 'xgboost']:
        p = models[name].predict_proba(X)
        probas.append(p)
    return np.concatenate(probas, axis=1)


def train_meta_learner(stacking_features_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform as sp_uniform

    param_dist = {
        'C': sp_uniform(0.01, 10),
    }

    print('  Tuning Logistic Regression meta-learner with RandomizedSearchCV (5 iters, 2-fold)...')
    base = LogisticRegression(
        solver='lbfgs',
        max_iter=1000, random_state=42
    )

    search = RandomizedSearchCV(
        base, param_dist, n_iter=5, cv=2,
        scoring='accuracy', random_state=42, n_jobs=1
    )
    search.fit(stacking_features_train, y_train)

    print(f'    Best C: {search.best_params_["C"]:.4f}')
    print(f'    Best CV accuracy: {search.best_score_:.4f}')
    return search.best_estimator_


def run_training():
    print('=' * 60)
    print('MODEL TRAINING (Stacking Ensemble + RandomizedSearchCV)')
    print('=' * 60)

    df, selected_features = load_data()
    print(f'Loaded {len(df)} rows, {len(selected_features)} selected features')

    train_df, test_df = chronological_split(df)
    print(f'Train: {len(train_df)} matches (2019-2024)')
    print(f'Test:  {len(test_df)} matches (2024-2025)')

    X_train = train_df[selected_features].fillna(0).values
    y_train = train_df['ftr_encoded'].values.astype(int)
    X_test = test_df[selected_features].fillna(0).values
    y_test = test_df['ftr_encoded'].values.astype(int)

    start = time.time()
    print('\n--- Hyperparameter Tuning & Training Base Models ---')
    models = {}
    models['catboost'] = tune_and_train_catboost(X_train, y_train)
    models['lightgbm'] = tune_and_train_lightgbm(X_train, y_train)
    models['xgboost'] = tune_and_train_xgboost(X_train, y_train)
    elapsed = time.time() - start
    print(f'\n  All base models tuned and trained in {elapsed:.1f}s')

    print('\n--- Generating Stacking Features ---')
    stack_train = get_stacking_features(models, X_train)
    stack_test = get_stacking_features(models, X_test)
    print(f'  Train stacking features: {stack_train.shape}')
    print(f'  Test stacking features:  {stack_test.shape}')

    print('\n--- Tuning & Training Meta-Learner ---')
    meta_model = train_meta_learner(stack_train, y_train)

    print('\n--- Saving Models ---')
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(models['catboost'], os.path.join(MODELS_DIR, 'catboost_model.pkl'), compress=3)
    joblib.dump(models['lightgbm'], os.path.join(MODELS_DIR, 'lightgbm_model.pkl'), compress=3)
    joblib.dump(models['xgboost'], os.path.join(MODELS_DIR, 'xgboost_model.pkl'), compress=3)
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'logistic_meta_model.pkl'), compress=3)
    print('  All models saved to trained_models/')

    return models, meta_model, X_test, y_test, stack_test, test_df, selected_features


if __name__ == '__main__':
    run_training()
