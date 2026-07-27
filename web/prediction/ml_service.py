import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings

FEATURE_NAME_MAP = {
    'home_form_wins_5': 'Home Last 5 Wins',
    'home_form_draws_5': 'Home Last 5 Draws',
    'home_form_losses_5': 'Home Last 5 Losses',
    'home_form_points_5': 'Home Last 5 Points',
    'home_form_goals_scored_5': 'Home Last 5 Goals Scored',
    'home_form_goals_conceded_5': 'Home Last 5 Goals Conceded',
    'home_form_goal_diff_5': 'Home Last 5 Goal Difference',
    'home_form_avg_goals_5': 'Home Avg Goals per Game',
    'home_form_avg_shots_5': 'Home Avg Shots per Game',
    'home_form_avg_sot_5': 'Home Avg Shots on Target',
    'away_form_wins_5': 'Away Last 5 Wins',
    'away_form_draws_5': 'Away Last 5 Draws',
    'away_form_losses_5': 'Away Last 5 Losses',
    'away_form_points_5': 'Away Last 5 Points',
    'away_form_goals_scored_5': 'Away Last 5 Goals Scored',
    'away_form_goals_conceded_5': 'Away Last 5 Goals Conceded',
    'away_form_goal_diff_5': 'Away Last 5 Goal Difference',
    'away_form_avg_goals_5': 'Away Avg Goals per Game',
    'away_form_avg_shots_5': 'Away Avg Shots per Game',
    'away_form_avg_sot_5': 'Away Avg Shots on Target',
    'home_form_wins_3': 'Home Last 3 Wins',
    'home_form_draws_3': 'Home Last 3 Draws',
    'home_form_losses_3': 'Home Last 3 Losses',
    'home_form_points_3': 'Home Last 3 Points',
    'home_form_goals_scored_3': 'Home Last 3 Goals Scored',
    'home_form_goals_conceded_3': 'Home Last 3 Goals Conceded',
    'home_form_goal_diff_3': 'Home Last 3 Goal Difference',
    'home_form_avg_goals_3': 'Home Last 3 Avg Goals',
    'home_form_avg_shots_3': 'Home Last 3 Avg Shots',
    'home_form_avg_sot_3': 'Home Last 3 Avg Shots on Target',
    'away_form_wins_3': 'Away Last 3 Wins',
    'away_form_draws_3': 'Away Last 3 Draws',
    'away_form_losses_3': 'Away Last 3 Losses',
    'away_form_points_3': 'Away Last 3 Points',
    'away_form_goals_scored_3': 'Away Last 3 Goals Scored',
    'away_form_goals_conceded_3': 'Away Last 3 Goals Conceded',
    'away_form_goal_diff_3': 'Away Last 3 Goal Difference',
    'away_form_avg_goals_3': 'Away Last 3 Avg Goals',
    'away_form_avg_shots_3': 'Away Last 3 Avg Shots',
    'away_form_avg_sot_3': 'Away Last 3 Avg Shots on Target',
    'home_form_wins_10': 'Home Last 10 Wins',
    'home_form_draws_10': 'Home Last 10 Draws',
    'home_form_losses_10': 'Home Last 10 Losses',
    'home_form_points_10': 'Home Last 10 Points',
    'home_form_goals_scored_10': 'Home Last 10 Goals Scored',
    'home_form_goals_conceded_10': 'Home Last 10 Goals Conceded',
    'home_form_goal_diff_10': 'Home Last 10 Goal Difference',
    'home_form_avg_goals_10': 'Home Last 10 Avg Goals',
    'home_form_avg_shots_10': 'Home Last 10 Avg Shots',
    'home_form_avg_sot_10': 'Home Last 10 Avg Shots on Target',
    'away_form_wins_10': 'Away Last 10 Wins',
    'away_form_draws_10': 'Away Last 10 Draws',
    'away_form_losses_10': 'Away Last 10 Losses',
    'away_form_points_10': 'Away Last 10 Points',
    'away_form_goals_scored_10': 'Away Last 10 Goals Scored',
    'away_form_goals_conceded_10': 'Away Last 10 Goals Conceded',
    'away_form_goal_diff_10': 'Away Last 10 Goal Difference',
    'away_form_avg_goals_10': 'Away Last 10 Avg Goals',
    'away_form_avg_shots_10': 'Away Last 10 Avg Shots',
    'away_form_avg_sot_10': 'Away Last 10 Avg Shots on Target',
    'home_team_home_perf_win_rate': 'Home Venue Win Rate',
    'home_team_home_perf_avg_goals': 'Home Venue Avg Goals Scored',
    'home_team_home_perf_avg_conceded': 'Home Venue Avg Goals Conceded',
    'home_team_home_perf_avg_shots': 'Home Venue Avg Shots',
    'home_team_home_perf_avg_sot': 'Home Venue Avg Shots on Target',
    'away_team_away_perf_win_rate': 'Away Venue Win Rate',
    'away_team_away_perf_avg_goals': 'Away Venue Avg Goals Scored',
    'away_team_away_perf_avg_conceded': 'Away Venue Avg Goals Conceded',
    'away_team_away_perf_avg_shots': 'Away Venue Avg Shots',
    'away_team_away_perf_avg_sot': 'Away Venue Avg Shots on Target',
    'h2h_home_wins': 'Head-to-Head Home Wins',
    'h2h_away_wins': 'Head-to-Head Away Wins',
    'h2h_draws': 'Head-to-Head Draws',
    'h2h_home_goals': 'Head-to-Head Home Goals',
    'h2h_away_goals': 'Head-to-Head Away Goals',
    'h2h_goal_diff': 'Head-to-Head Goal Difference',
    'h2h_avg_total_goals': 'Head-to-Head Avg Total Goals',
    'h2h_home_scoring_rate': 'Head-to-Head Home Scoring Rate',
    'h2h_away_scoring_rate': 'Head-to-Head Away Scoring Rate',
    'h2h_avg_shots': 'Head-to-Head Avg Shots',
    'h2h_avg_corners': 'Head-to-Head Avg Corners',
    'h2h_home_dominance': 'Head-to-Head Home Dominance',
    'home_atk_avg_goals': 'Home Avg Goals (Overall)',
    'home_atk_avg_shots': 'Home Avg Shots (Overall)',
    'home_atk_avg_sot': 'Home Avg Shots on Target (Overall)',
    'home_atk_avg_corners': 'Home Avg Corners (Overall)',
    'away_atk_avg_goals': 'Away Avg Goals (Overall)',
    'away_atk_avg_shots': 'Away Avg Shots (Overall)',
    'away_atk_avg_sot': 'Away Avg Shots on Target (Overall)',
    'away_atk_avg_corners': 'Away Avg Corners (Overall)',
    'home_def_avg_conceded': 'Home Avg Goals Conceded',
    'home_def_clean_sheet_rate': 'Home Clean Sheet Rate',
    'home_def_shots_faced': 'Home Avg Shots Faced',
    'home_def_avg_fouls': 'Home Avg Fouls Committed',
    'home_def_avg_yellow': 'Home Avg Yellow Cards',
    'home_def_avg_red': 'Home Avg Red Cards',
    'away_def_avg_conceded': 'Away Avg Goals Conceded',
    'away_def_clean_sheet_rate': 'Away Clean Sheet Rate',
    'away_def_shots_faced': 'Away Avg Shots Faced',
    'away_def_avg_fouls': 'Away Avg Fouls Committed',
    'away_def_avg_yellow': 'Away Avg Yellow Cards',
    'away_def_avg_red': 'Away Avg Red Cards',
    'home_mom_win_streak': 'Home Current Win Streak',
    'home_mom_loss_streak': 'Home Current Loss Streak',
    'home_mom_unbeaten_streak': 'Home Current Unbeaten Streak',
    'away_mom_win_streak': 'Away Current Win Streak',
    'away_mom_loss_streak': 'Away Current Loss Streak',
    'away_mom_unbeaten_streak': 'Away Current Unbeaten Streak',
    'home_eff_goal_conversion': 'Home Goal Conversion Rate',
    'home_eff_shot_accuracy': 'Home Shot Accuracy',
    'home_eff_goals_per_sot': 'Home Goals per Shot on Target',
    'home_eff_sot_conversion': 'Home Shots on Target Rate',
    'away_eff_goal_conversion': 'Away Goal Conversion Rate',
    'away_eff_shot_accuracy': 'Away Shot Accuracy',
    'away_eff_goals_per_sot': 'Away Goals per Shot on Target',
    'away_eff_sot_conversion': 'Away Shots on Target Rate',
    'home_dom_shots_diff': 'Home Shots Dominance',
    'home_dom_sot_diff': 'Home Shots on Target Dominance',
    'home_dom_corners_diff': 'Home Corner Dominance',
    'home_dom_goals_diff': 'Home Goal Dominance',
    'away_dom_shots_diff': 'Away Shots Dominance',
    'away_dom_sot_diff': 'Away Shots on Target Dominance',
    'away_dom_corners_diff': 'Away Corner Dominance',
    'away_dom_goals_diff': 'Away Goal Dominance',
    'home_trend_goals': 'Home Goals Trend',
    'home_trend_conceded': 'Home Conceded Trend',
    'home_trend_shots': 'Home Shots Trend',
    'home_trend_sot': 'Home Shots on Target Trend',
    'away_trend_goals': 'Away Goals Trend',
    'away_trend_conceded': 'Away Conceded Trend',
    'away_trend_shots': 'Away Shots Trend',
    'away_trend_sot': 'Away Shots on Target Trend',
    'home_consist_goals_std': 'Home Goals Consistency',
    'home_consist_conceded_std': 'Home Conceded Consistency',
    'home_consist_shots_std': 'Home Shots Consistency',
    'home_consist_sot_std': 'Home Shots on Target Consistency',
    'home_consist_corners_std': 'Home Corners Consistency',
    'home_consist_gd_std': 'Home Goal Difference Consistency',
    'away_consist_goals_std': 'Away Goals Consistency',
    'away_consist_conceded_std': 'Away Conceded Consistency',
    'away_consist_shots_std': 'Away Shots Consistency',
    'away_consist_sot_std': 'Away Shots on Target Consistency',
    'away_consist_corners_std': 'Away Corners Consistency',
    'away_consist_gd_std': 'Away Goal Difference Consistency',
    'home_ht_goals_scored_avg': 'Home Half-Time Goals Scored',
    'home_ht_goals_conceded_avg': 'Home Half-Time Goals Conceded',
    'home_ht_win_rate': 'Home Half-Time Win Rate',
    'home_ht_draw_rate': 'Home Half-Time Draw Rate',
    'away_ht_goals_scored_avg': 'Away Half-Time Goals Scored',
    'away_ht_goals_conceded_avg': 'Away Half-Time Goals Conceded',
    'away_ht_win_rate': 'Away Half-Time Win Rate',
    'away_ht_draw_rate': 'Away Half-Time Draw Rate',
    'home_disc_fouls': 'Home Avg Fouls',
    'home_disc_yellow': 'Home Avg Yellow Cards',
    'home_disc_red': 'Home Avg Red Cards',
    'away_disc_fouls': 'Away Avg Fouls',
    'away_disc_yellow': 'Away Avg Yellow Cards',
    'away_disc_red': 'Away Avg Red Cards',
    'home_team_fd_home_opp_win_rate': 'Home Opponent Win Rate',
    'home_team_fd_home_opp_avg_gd': 'Home Opponent Avg Goal Diff',
    'home_team_fd_home_opp_avg_points': 'Home Opponent Avg Points',
    'away_team_fd_away_opp_win_rate': 'Away Opponent Win Rate',
    'away_team_fd_away_opp_avg_gd': 'Away Opponent Avg Goal Diff',
    'away_team_fd_away_opp_avg_points': 'Away Opponent Avg Points',
    'home_team_venue_mom_home_unbeaten': 'Home Unbeaten Run at Home',
    'home_team_venue_mom_home_winning': 'Home Winning Run at Home',
    'home_team_venue_mom_home_scoring': 'Home Scoring Run at Home',
    'away_team_venue_mom_away_unbeaten': 'Away Unbeaten Run Away',
    'away_team_venue_mom_away_winning': 'Away Winning Run Away',
    'away_team_venue_mom_away_scoring': 'Away Scoring Run Away',
    'home_team_win_pct': 'Home Overall Win Percentage',
    'home_team_ppm': 'Home Points per Match',
    'away_team_win_pct': 'Away Overall Win Percentage',
    'away_team_ppm': 'Away Points per Match',
    'home_team_cs_rate': 'Home Clean Sheet Rate',
    'away_team_cs_rate': 'Away Clean Sheet Rate',
    'home_elo': 'Home ELO Rating',
    'away_elo': 'Away ELO Rating',
    'elo_diff': 'ELO Rating Difference',
    'home_xg_proxy': 'Home Expected Goals (Proxy)',
    'away_xg_proxy': 'Away Expected Goals (Proxy)',
    'xg_proxy_diff': 'Expected Goals Difference',
    'diff_form_points_3': 'Form Points Diff (3 games)',
    'diff_form_goals_scored_3': 'Goals Scored Diff (3 games)',
    'diff_form_avg_shots_3': 'Avg Shots Diff (3 games)',
    'diff_form_avg_sot_3': 'Avg Shots on Target Diff (3 games)',
    'diff_form_goal_diff_3': 'Goal Difference Diff (3 games)',
    'diff_form_points_5': 'Form Points Diff (5 games)',
    'diff_form_goals_scored_5': 'Goals Scored Diff (5 games)',
    'diff_form_avg_shots_5': 'Avg Shots Diff (5 games)',
    'diff_form_avg_sot_5': 'Avg Shots on Target Diff (5 games)',
    'diff_form_goal_diff_5': 'Goal Difference Diff (5 games)',
    'diff_form_points_10': 'Form Points Diff (10 games)',
    'diff_form_goals_scored_10': 'Goals Scored Diff (10 games)',
    'diff_form_avg_shots_10': 'Avg Shots Diff (10 games)',
    'diff_form_avg_sot_10': 'Avg Shots on Target Diff (10 games)',
    'diff_form_goal_diff_10': 'Goal Difference Diff (10 games)',
    'home_advantage_index': 'Home Advantage Index',
}


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

            shap_features = self._get_feature_importance(features)
            last_5_home = self._get_last_5_games(fe, home_team)
            last_5_away = self._get_last_5_games(fe, away_team)

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
                'last_5_home': last_5_home,
                'last_5_away': last_5_away,
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
            'last_5_home': [],
            'last_5_away': [],
        }

    def _get_last_5_games(self, fe, team):
        from ml_engine.feature_engineering import _get_team_all_sorted, _get_team_result
        all_matches = _get_team_all_sorted(fe.dataset, team).tail(5)
        results = []
        for _, row in all_matches.iterrows():
            result, gf, ga = _get_team_result(row, team)
            opponent = row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam']
            venue = 'H' if row['HomeTeam'] == team else 'A'
            results.append({
                'opponent': opponent,
                'venue': venue,
                'result': result,
                'goals_for': int(gf),
                'goals_against': int(ga),
                'score': f'{int(gf)}-{int(ga)}',
            })
        return results

    def _get_shap_features(self, features):
        return []

    def _get_feature_importance(self, features):
        try:
            importances = self.xgboost_model.feature_importances_
            feat_names = list(self.selected_features)
            pairs = list(zip(feat_names, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top = pairs[:5]
            return [
                {
                    'name': FEATURE_NAME_MAP.get(name, name),
                    'importance': round(float(imp), 4),
                }
                for name, imp in top if imp > 0
            ]
        except Exception:
            return []
