from django.shortcuts import render
import os
import pandas as pd
from django.conf import settings


def features(request):
    context = {
        'feature_groups': [],
        'total_raw': 198,
        'total_filtered': 157,
        'total_selected': 50,
    }

    try:
        importance_path = settings.RESULTS_DIR / 'tables' / 'feature_importance.csv'
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            top_features = importance_df.head(20).to_dict('records')
            context['top_features'] = top_features
        else:
            context['top_features'] = []
    except Exception:
        context['top_features'] = []

    try:
        selected_path = settings.ML_MODELS_DIR / 'selected_features.pkl'
        if selected_path.exists():
            import joblib
            selected = joblib.load(selected_path)
            context['selected_features'] = selected
        else:
            context['selected_features'] = []
    except Exception:
        context['selected_features'] = []

    context['feature_groups'] = [
        {
            'name': 'Team Form (Windows: 3, 5, 10)',
            'description': 'Wins, draws, losses, points, goals scored/conceded, goal difference, avg goals/shots/sot over rolling windows.',
            'count': 60,
            'examples': ['home_form_points_10', 'away_form_goals_scored_5'],
        },
        {
            'name': 'Venue Performance',
            'description': 'Win rate, avg goals, avg conceded, avg shots, avg sot for home/away venue.',
            'count': 10,
            'examples': ['home_team_home_perf_win_rate', 'away_team_away_perf_avg_shots'],
        },
        {
            'name': 'Head-to-Head',
            'description': 'Historical matchups: wins, draws, goals, scoring rates, shot/corner totals, dominance.',
            'count': 12,
            'examples': ['h2h_home_wins', 'h2h_away_scoring_rate', 'h2h_home_dominance'],
        },
        {
            'name': 'Attack / Defense / Discipline',
            'description': 'Avg goals, shots, sot, corners, conceded, clean sheet rate, fouls, yellows, reds.',
            'count': 36,
            'examples': ['home_atk_avg_goals', 'away_def_clean_sheet_rate'],
        },
        {
            'name': 'Elo Ratings',
            'description': 'Elo computed from full match history. Home, away, and difference.',
            'count': 3,
            'examples': ['home_elo', 'away_elo', 'elo_diff'],
        },
        {
            'name': 'Momentum & Streaks',
            'description': 'Win streak, loss streak, unbeaten streak for both teams.',
            'count': 6,
            'examples': ['home_mom_win_streak', 'away_mom_unbeaten_streak'],
        },
        {
            'name': 'Efficiency & Dominance',
            'description': 'Goal conversion, shot accuracy, shots/sot/corners/goals difference.',
            'count': 16,
            'examples': ['home_eff_goal_conversion', 'away_dom_shots_diff'],
        },
        {
            'name': 'Trend & Consistency',
            'description': 'Linear trend slopes and std dev for goals, conceded, shots, sot, corners, goal diff.',
            'count': 24,
            'examples': ['home_trend_goals', 'away_consist_shots_std'],
        },
        {
            'name': 'Half-Time Performance',
            'description': 'HT goals scored, conceded, win rate, draw rate.',
            'count': 8,
            'examples': ['home_ht_win_rate', 'home_ht_goals_scored_avg'],
        },
        {
            'name': 'Fixture Difficulty & Venue Momentum',
            'description': 'Opponent strength (win rate, avg GD, avg points), unbeaten/winning/scoring streaks at venue.',
            'count': 12,
            'examples': ['home_team_fd_home_opp_win_rate', 'away_team_venue_mom_away_unbeaten'],
        },
        {
            'name': 'xG Proxy & Derived',
            'description': 'xG proxy from sot, home advantage index, difference features.',
            'count': 11,
            'examples': ['home_xg_proxy', 'xg_proxy_diff', 'home_advantage_index'],
        },
    ]

    return render(request, 'ml_engine/features.html', context)
