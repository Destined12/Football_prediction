import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings


class PredictionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self.xgboost_model = None
        self.svm_model = None
        self.rf_model = None
        self.meta_model = None
        self.selected_features = None
        self.dataset = None
        self._loaded = False

    def load_models(self):
        models_dir = settings.ML_MODELS_DIR
        if not models_dir.exists():
            return False

        try:
            self.xgboost_model = joblib.load(models_dir / 'xgboost_model.pkl')
            self.svm_model = joblib.load(models_dir / 'svm_pipeline.pkl')
            self.rf_model = joblib.load(models_dir / 'random_forest_model.pkl')
            self.meta_model = joblib.load(models_dir / 'logistic_meta_model.pkl')
            self.selected_features = joblib.load(models_dir / 'selected_features.pkl')
            self._loaded = True
            return True
        except Exception:
            return False

    def predict(self, home_team, away_team):
        if not self._loaded:
            loaded = self.load_models()
            if not loaded:
                return self._fallback_prediction(home_team, away_team)

        try:
            from ml_engine.feature_engineering import FeatureEngineering
            fe = FeatureEngineering()
            features = fe.compute_match_features(home_team, away_team)

            if features is None:
                return self._fallback_prediction(home_team, away_team)

            selected = features[self.selected_features].fillna(0).values.reshape(1, -1)

            xgb_proba = self.xgboost_model.predict_proba(selected)[0]
            svm_proba = self.svm_model.predict_proba(selected)[0]
            rf_proba = self.rf_model.predict_proba(selected)[0]

            meta_features = np.concatenate([xgb_proba, svm_proba, rf_proba]).reshape(1, -1)
            final_proba = self.meta_model.predict_proba(meta_features)[0]

            labels = {0: 'H', 1: 'D', 2: 'A'}
            label_names = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
            predicted_class = int(np.argmax(final_proba))
            confidence = float(final_proba[predicted_class])

            shap_features = self._get_shap_features(features)

            return {
                'predicted_result': labels[predicted_class],
                'predicted_label': label_names[predicted_class],
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    'home': round(float(final_proba[0]) * 100, 2),
                    'draw': round(float(final_proba[1]) * 100, 2),
                    'away': round(float(final_proba[2]) * 100, 2),
                },
                'shap_features': shap_features,
            }
        except Exception:
            return self._fallback_prediction(home_team, away_team)

    def _fallback_prediction(self, home_team, away_team):
        return {
            'predicted_result': 'H',
            'predicted_label': 'Home Win',
            'confidence': 45.0,
            'probabilities': {'home': 45.0, 'draw': 25.0, 'away': 30.0},
            'shap_features': [],
        }

    def _get_shap_features(self, features):
        try:
            import shap
            explainer = shap.TreeExplainer(self.xgboost_model)
            shap_values = explainer.shap_values(features[self.selected_features].fillna(0))

            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            if mean_shap.ndim > 1:
                mean_shap = mean_shap.mean(axis=1)

            top_indices = np.argsort(mean_shap)[-5:][::-1]
            result = []
            for idx in top_indices:
                result.append({
                    'name': self.selected_features[idx],
                    'value': round(float(features[self.selected_features].iloc[0, idx]), 4),
                    'importance': round(float(mean_shap[idx]), 4),
                })
            return result
        except Exception:
            return []
