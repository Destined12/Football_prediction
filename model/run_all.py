import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.feature_engineering import run_feature_engineering
from model.feature_selection import run_feature_selection
from model.train_step import load_data, chronological_split
from model.eval_step import run_evaluation
from model.shap_step import run_shap_explainability


def run_all():
    total_start = time.time()

    print('\n' + '=' * 60)
    print('STEP 1: FEATURE ENGINEERING')
    print('=' * 60)
    run_feature_engineering()

    print('\n' + '=' * 60)
    print('STEP 2: FEATURE SELECTION')
    print('=' * 60)
    run_feature_selection()

    print('\n' + '=' * 60)
    print('STEP 3: MODEL TRAINING')
    print('=' * 60)
    from model.train_step import train_xgboost, train_svm, train_random_forest, train_meta_learner
    import joblib

    MODELS_DIR = os.path.join(PROJECT_ROOT, 'web', 'trained_models')

    df, selected_features = load_data()
    train_df, test_df = chronological_split(df)
    X_train = train_df[selected_features].fillna(0).values
    y_train = train_df['ftr_encoded'].values.astype(int)
    X_test = test_df[selected_features].fillna(0).values
    y_test = test_df['ftr_encoded'].values.astype(int)

    print(f'  Data: {len(train_df)} train, {len(test_df)} test, {len(selected_features)} features')

    os.makedirs(MODELS_DIR, exist_ok=True)

    print('\n  --- XGBoost ---')
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'), compress=3)
    print(f'  Saved ({time.time()-t0:.1f}s)')

    print('\n  --- SVM ---')
    t0 = time.time()
    svm_model = train_svm(X_train, y_train)
    joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_pipeline.pkl'), compress=3)
    print(f'  Saved ({time.time()-t0:.1f}s)')

    print('\n  --- Random Forest ---')
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.pkl'), compress=3)
    print(f'  Saved ({time.time()-t0:.1f}s)')

    print('\n  --- Meta-Learner ---')
    t0 = time.time()
    meta_model = train_meta_learner([xgb_model, svm_model, rf_model], X_train, y_train)
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'logistic_meta_model.pkl'), compress=3)
    print(f'  Saved ({time.time()-t0:.1f}s)')

    print('\n' + '=' * 60)
    print('STEP 4: EVALUATION')
    print('=' * 60)
    run_evaluation()

    print('\n' + '=' * 60)
    print('STEP 5: SHAP EXPLAINABILITY')
    print('=' * 60)
    run_shap_explainability()

    total = time.time() - total_start
    print(f'\n{"=" * 60}')
    print(f'ALL STEPS COMPLETE ({total:.1f}s)')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    run_all()
