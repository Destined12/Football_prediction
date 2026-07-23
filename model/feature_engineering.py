import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_dataset.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

FTR_MAP = {'H': 0, 'D': 1, 'A': 2}

SUPPORTED_CLUBS = [
    'Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United'
]

TEAM_NAME_MAP = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Wolves': 'Wolverhampton',
    "Nott'm Forest": 'Nottingham Forest',
    "Spurs": 'Tottenham',
}


def load_cleaned_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df['HomeTeam'] = df['HomeTeam'].replace(TEAM_NAME_MAP)
    df['AwayTeam'] = df['AwayTeam'].replace(TEAM_NAME_MAP)
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def get_team_history(df, team, before_idx):
    hist = df.iloc[:before_idx]
    home = hist[hist['HomeTeam'] == team].copy()
    away = hist[hist['AwayTeam'] == team].copy()
    return home, away, hist


def get_team_all_sorted(df, team, before_idx):
    home, away, hist = get_team_history(df, team, before_idx)
    all_matches = pd.concat([home, away]).sort_values('Date')
    return all_matches


def _safe_mean(values):
    if len(values) == 0:
        return 0
    return np.mean(values)


def _safe_std(values):
    if len(values) < 2:
        return 0
    return np.std(values, ddof=1)


def _linear_trend(values, window=5):
    if len(values) < window:
        return 0
    vals = values[-window:]
    x = np.arange(window, dtype=float)
    slope = np.polyfit(x, vals, 1)[0]
    return slope


def _streak(results, target):
    count = 0
    for r in reversed(results):
        if r == target:
            count += 1
        else:
            break
    return count


def _unbeaten_streak(results):
    count = 0
    for r in reversed(results):
        if r in ('W', 'D'):
            count += 1
        else:
            break
    return count


def _get_team_result(row, team):
    if row['HomeTeam'] == team:
        if row['FTR'] == 'H':
            return 'W', row['FTHG'], row['FTAG']
        elif row['FTR'] == 'D':
            return 'D', row['FTHG'], row['FTAG']
        else:
            return 'L', row['FTHG'], row['FTAG']
    else:
        if row['FTR'] == 'A':
            return 'W', row['FTAG'], row['FTHG']
        elif row['FTR'] == 'D':
            return 'D', row['FTAG'], row['FTHG']
        else:
            return 'L', row['FTAG'], row['FTHG']


def _get_team_stats(row, team):
    if row['HomeTeam'] == team:
        return {
            'goals_for': row['FTHG'], 'goals_against': row['FTAG'],
            'shots': row['HS'], 'shots_against': row['AS'],
            'sot': row['HST'], 'sot_against': row['AST'],
            'corners': row['HC'], 'corners_against': row['AC'],
            'fouls': row['HF'], 'yellows': row['HY'], 'reds': row['HR'],
            'ht_goals_for': row['HTHG'], 'ht_goals_against': row['HTAG'],
            'venue': 'home',
        }
    else:
        return {
            'goals_for': row['FTAG'], 'goals_against': row['FTHG'],
            'shots': row['AS'], 'shots_against': row['HS'],
            'sot': row['AST'], 'sot_against': row['HST'],
            'corners': row['AC'], 'corners_against': row['HC'],
            'fouls': row['AF'], 'yellows': row['AY'], 'reds': row['AR'],
            'ht_goals_for': row['HTAG'], 'ht_goals_against': row['HTHG'],
            'venue': 'away',
        }


# ---------------------------------------------------------------------------
# ELO RATING SYSTEM
# ---------------------------------------------------------------------------

def compute_elo_ratings(df):
    elo = {}
    K = 20
    INIT = 1500

    home_elos = []
    away_elos = []

    for i in range(len(df)):
        ht = df.iloc[i]['HomeTeam']
        at = df.iloc[i]['AwayTeam']
        h_elo = elo.get(ht, INIT)
        a_elo = elo.get(at, INIT)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        expected_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        expected_a = 1 - expected_h

        ftr = df.iloc[i]['FTR']
        if ftr == 'H':
            actual_h, actual_a = 1.0, 0.0
        elif ftr == 'D':
            actual_h, actual_a = 0.5, 0.5
        else:
            actual_h, actual_a = 0.0, 1.0

        elo[ht] = h_elo + K * (actual_h - expected_h)
        elo[at] = a_elo + K * (actual_a - expected_a)

    df = df.copy()
    df['_home_elo'] = home_elos
    df['_away_elo'] = away_elos
    return df


# ---------------------------------------------------------------------------
# GROUP 1: TEAM FORM (Multi-Window)
# ---------------------------------------------------------------------------

def compute_team_form_for_team(df, team, before_idx, window=5):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {k: 0 for k in [
            f'form_wins_{window}', f'form_draws_{window}', f'form_losses_{window}',
            f'form_points_{window}', f'form_goals_scored_{window}',
            f'form_goals_conceded_{window}', f'form_goal_diff_{window}',
            f'form_avg_goals_{window}', f'form_avg_shots_{window}',
            f'form_avg_sot_{window}',
        ]}

    wins, draws, losses = 0, 0, 0
    gs, gc, shots, sot = 0, 0, 0, 0

    for _, row in all_matches.iterrows():
        result, gf, ga = _get_team_result(row, team)
        stats = _get_team_stats(row, team)
        if result == 'W':
            wins += 1
        elif result == 'D':
            draws += 1
        else:
            losses += 1
        gs += gf
        gc += ga
        shots += stats['shots']
        sot += stats['sot']

    n = len(all_matches)
    return {
        f'form_wins_{window}': wins,
        f'form_draws_{window}': draws,
        f'form_losses_{window}': losses,
        f'form_points_{window}': wins * 3 + draws,
        f'form_goals_scored_{window}': gs,
        f'form_goals_conceded_{window}': gc,
        f'form_goal_diff_{window}': gs - gc,
        f'form_avg_goals_{window}': gs / n,
        f'form_avg_shots_{window}': shots / n,
        f'form_avg_sot_{window}': sot / n,
    }


# ---------------------------------------------------------------------------
# GROUP 2/3: HOME/AWAY PERFORMANCE (Last 10 at venue)
# ---------------------------------------------------------------------------

def compute_venue_performance(df, team, venue, before_idx, window=10):
    hist = df.iloc[:before_idx]
    if venue == 'home':
        matches = hist[hist['HomeTeam'] == team].tail(window)
        goal_for_col, goal_against_col = 'FTHG', 'FTAG'
        shots_col, sot_col = 'HS', 'HST'
        result_col = 'FTR'
        win_val = 'H'
    else:
        matches = hist[hist['AwayTeam'] == team].tail(window)
        goal_for_col, goal_against_col = 'FTAG', 'FTHG'
        shots_col, sot_col = 'AS', 'AST'
        result_col = 'FTR'
        win_val = 'A'

    prefix = f'{venue}_perf'
    if len(matches) == 0:
        return {k: 0 for k in [
            f'{prefix}_win_rate', f'{prefix}_avg_goals', f'{prefix}_avg_conceded',
            f'{prefix}_avg_shots', f'{prefix}_avg_sot',
        ]}

    wins = (matches[result_col] == win_val).sum()
    n = len(matches)
    return {
        f'{prefix}_win_rate': wins / n,
        f'{prefix}_avg_goals': matches[goal_for_col].mean(),
        f'{prefix}_avg_conceded': matches[goal_against_col].mean(),
        f'{prefix}_avg_shots': matches[shots_col].mean(),
        f'{prefix}_avg_sot': matches[sot_col].mean(),
    }


# ---------------------------------------------------------------------------
# GROUP 4: HEAD-TO-HEAD (Expanded)
# ---------------------------------------------------------------------------

def compute_head_to_head(df, home_team, away_team, before_idx, window=10):
    hist = df.iloc[:before_idx]
    h2h = hist[
        ((hist['HomeTeam'] == home_team) & (hist['AwayTeam'] == away_team)) |
        ((hist['HomeTeam'] == away_team) & (hist['AwayTeam'] == home_team))
    ].tail(window)

    keys = [
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
        'h2h_avg_total_goals', 'h2h_home_scoring_rate', 'h2h_away_scoring_rate',
        'h2h_avg_shots', 'h2h_avg_corners', 'h2h_home_dominance',
    ]
    if len(h2h) == 0:
        return {k: 0 for k in keys}

    hw, aw, dr = 0, 0, 0
    hg, ag = 0, 0
    total_shots, total_corners = 0, 0

    for _, row in h2h.iterrows():
        if row['HomeTeam'] == home_team:
            hg += row['FTHG']
            ag += row['FTAG']
            total_shots += row['HS'] + row['AS']
            total_corners += row['HC'] + row['AC']
            if row['FTR'] == 'H':
                hw += 1
            elif row['FTR'] == 'D':
                dr += 1
            else:
                aw += 1
        else:
            hg += row['FTAG']
            ag += row['FTHG']
            total_shots += row['HS'] + row['AS']
            total_corners += row['HC'] + row['AC']
            if row['FTR'] == 'A':
                hw += 1
            elif row['FTR'] == 'D':
                dr += 1
            else:
                aw += 1

    n = len(h2h)
    return {
        'h2h_home_wins': hw,
        'h2h_away_wins': aw,
        'h2h_draws': dr,
        'h2h_home_goals': hg,
        'h2h_away_goals': ag,
        'h2h_goal_diff': hg - ag,
        'h2h_avg_total_goals': (hg + ag) / n,
        'h2h_home_scoring_rate': hg / n,
        'h2h_away_scoring_rate': ag / n,
        'h2h_avg_shots': total_shots / n,
        'h2h_avg_corners': total_corners / n,
        'h2h_home_dominance': (hw - aw) / n,
    }


# ---------------------------------------------------------------------------
# GROUP 5: ATTACKING STRENGTH
# ---------------------------------------------------------------------------

def compute_attacking_strength(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {k: 0 for k in [
            'atk_avg_goals', 'atk_avg_shots', 'atk_avg_sot', 'atk_avg_corners',
        ]}

    goals, shots, sot, corners = 0, 0, 0, 0
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        goals += stats['goals_for']
        shots += stats['shots']
        sot += stats['sot']
        corners += stats['corners']

    n = len(all_matches)
    return {
        'atk_avg_goals': goals / n,
        'atk_avg_shots': shots / n,
        'atk_avg_sot': sot / n,
        'atk_avg_corners': corners / n,
    }


# ---------------------------------------------------------------------------
# GROUP 6: DEFENSIVE STRENGTH
# ---------------------------------------------------------------------------

def compute_defensive_strength(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {k: 0 for k in [
            'def_avg_conceded', 'def_clean_sheet_rate', 'def_shots_faced',
            'def_avg_fouls', 'def_avg_yellow', 'def_avg_red',
        ]}

    gc, cs, sf, fouls, yellows, reds = 0, 0, 0, 0, 0, 0
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        gc += stats['goals_against']
        if stats['goals_against'] == 0:
            cs += 1
        sf += stats['shots_against']
        fouls += stats['fouls']
        yellows += stats['yellows']
        reds += stats['reds']

    n = len(all_matches)
    return {
        'def_avg_conceded': gc / n,
        'def_clean_sheet_rate': cs / n,
        'def_shots_faced': sf / n,
        'def_avg_fouls': fouls / n,
        'def_avg_yellow': yellows / n,
        'def_avg_red': reds / n,
    }


# ---------------------------------------------------------------------------
# GROUP 7: DISCIPLINE
# ---------------------------------------------------------------------------

def compute_discipline(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {'disc_fouls': 0, 'disc_yellow': 0, 'disc_red': 0}

    fouls, yellows, reds = 0, 0, 0
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        fouls += stats['fouls']
        yellows += stats['yellows']
        reds += stats['reds']

    n = len(all_matches)
    return {
        'disc_fouls': fouls / n,
        'disc_yellow': yellows / n,
        'disc_red': reds / n,
    }


# ---------------------------------------------------------------------------
# GROUP 8: MOMENTUM
# ---------------------------------------------------------------------------

def compute_momentum(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {'mom_win_streak': 0, 'mom_loss_streak': 0, 'mom_unbeaten_streak': 0}

    results = []
    for _, row in all_matches.iterrows():
        result, _, _ = _get_team_result(row, team)
        results.append(result)

    return {
        'mom_win_streak': _streak(results, 'W'),
        'mom_loss_streak': _streak(results, 'L'),
        'mom_unbeaten_streak': _unbeaten_streak(results),
    }


# ---------------------------------------------------------------------------
# GROUP 9: GOAL EFFICIENCY
# ---------------------------------------------------------------------------

def compute_goal_efficiency(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    prefix = 'eff'
    keys = [f'{prefix}_goal_conversion', f'{prefix}_shot_accuracy',
            f'{prefix}_goals_per_sot', f'{prefix}_sot_conversion']
    if len(all_matches) == 0:
        return {k: 0 for k in keys}

    goals, shots, sot = 0, 0, 0
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        goals += stats['goals_for']
        shots += stats['shots']
        sot += stats['sot']

    return {
        f'{prefix}_goal_conversion': goals / shots if shots > 0 else 0,
        f'{prefix}_shot_accuracy': sot / shots if shots > 0 else 0,
        f'{prefix}_goals_per_sot': goals / sot if sot > 0 else 0,
        f'{prefix}_sot_conversion': sot / shots if shots > 0 else 0,
    }


# ---------------------------------------------------------------------------
# GROUP 10: MATCH DOMINANCE (Differentials over last 10)
# ---------------------------------------------------------------------------

def compute_match_dominance(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    prefix = 'dom'
    keys = [f'{prefix}_shots_diff', f'{prefix}_sot_diff',
            f'{prefix}_corners_diff', f'{prefix}_goals_diff']
    if len(all_matches) == 0:
        return {k: 0 for k in keys}

    shot_diffs, sot_diffs, corner_diffs, goal_diffs = [], [], [], []
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        shot_diffs.append(stats['shots'] - stats['shots_against'])
        sot_diffs.append(stats['sot'] - stats['sot_against'])
        corner_diffs.append(stats['corners'] - stats['corners_against'])
        goal_diffs.append(stats['goals_for'] - stats['goals_against'])

    return {
        f'{prefix}_shots_diff': _safe_mean(shot_diffs),
        f'{prefix}_sot_diff': _safe_mean(sot_diffs),
        f'{prefix}_corners_diff': _safe_mean(corner_diffs),
        f'{prefix}_goals_diff': _safe_mean(goal_diffs),
    }


# ---------------------------------------------------------------------------
# GROUP 11: RECENT TREND (Linear slope over last 5)
# ---------------------------------------------------------------------------

def compute_recent_trend(df, team, before_idx, window=5):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    prefix = 'trend'
    keys = [f'{prefix}_goals', f'{prefix}_conceded',
            f'{prefix}_shots', f'{prefix}_sot']
    if len(all_matches) < 3:
        return {k: 0 for k in keys}

    goals_list, conceded_list, shots_list, sot_list = [], [], [], []
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        goals_list.append(stats['goals_for'])
        conceded_list.append(stats['goals_against'])
        shots_list.append(stats['shots'])
        sot_list.append(stats['sot'])

    return {
        f'{prefix}_goals': _linear_trend(goals_list, window),
        f'{prefix}_conceded': _linear_trend(conceded_list, window),
        f'{prefix}_shots': _linear_trend(shots_list, window),
        f'{prefix}_sot': _linear_trend(sot_list, window),
    }


# ---------------------------------------------------------------------------
# GROUP 12: FIXTURE DIFFICULTY
# ---------------------------------------------------------------------------

def compute_fixture_difficulty(df, team, venue, before_idx, window=10):
    hist = df.iloc[:before_idx]
    if venue == 'home':
        matches = hist[hist['HomeTeam'] == team].tail(window)
        opp_col = 'AwayTeam'
    else:
        matches = hist[hist['AwayTeam'] == team].tail(window)
        opp_col = 'HomeTeam'

    prefix = f'fd_{venue}'
    keys = [f'{prefix}_opp_win_rate', f'{prefix}_opp_avg_gd', f'{prefix}_opp_avg_points']
    if len(matches) == 0:
        return {k: 0 for k in keys}

    opp_win_rates, opp_gds, opp_points = [], [], []
    for _, row in matches.iterrows():
        opp = row[opp_col]
        opp_hist = hist[hist['Date'] < row['Date']]
        opp_home = opp_hist[opp_hist['HomeTeam'] == opp]
        opp_away = opp_hist[opp_hist['AwayTeam'] == opp]
        opp_all = pd.concat([opp_home, opp_away])

        if len(opp_all) < 3:
            opp_win_rates.append(0.33)
            opp_gds.append(0)
            opp_points.append(1.0)
            continue

        opp_wins = ((opp_all['HomeTeam'] == opp) & (opp_all['FTR'] == 'H')).sum() + \
                   ((opp_all['AwayTeam'] == opp) & (opp_all['FTR'] == 'A')).sum()
        opp_draws = (opp_all['FTR'] == 'D').sum()
        opp_n = len(opp_all)

        opp_gs = 0
        opp_ga = 0
        for _, orow in opp_all.iterrows():
            if orow['HomeTeam'] == opp:
                opp_gs += orow['FTHG']
                opp_ga += orow['FTAG']
            else:
                opp_gs += orow['FTAG']
                opp_ga += orow['FTHG']

        opp_win_rates.append(opp_wins / opp_n)
        opp_gds.append((opp_gs - opp_ga) / opp_n)
        opp_points.append((opp_wins * 3 + opp_draws) / opp_n)

    return {
        f'{prefix}_opp_win_rate': _safe_mean(opp_win_rates),
        f'{prefix}_opp_avg_gd': _safe_mean(opp_gds),
        f'{prefix}_opp_avg_points': _safe_mean(opp_points),
    }


# ---------------------------------------------------------------------------
# GROUP 13: HOME/AWAY MOMENTUM (Venue-specific streaks)
# ---------------------------------------------------------------------------

def compute_venue_momentum(df, team, venue, before_idx, window=20):
    hist = df.iloc[:before_idx]
    if venue == 'home':
        matches = hist[hist['HomeTeam'] == team].tail(window)
    else:
        matches = hist[hist['AwayTeam'] == team].tail(window)

    prefix = f'venue_mom_{venue}'
    keys = [f'{prefix}_unbeaten', f'{prefix}_winning', f'{prefix}_scoring']
    if len(matches) == 0:
        return {k: 0 for k in keys}

    results, scored = [], []
    for _, row in matches.iterrows():
        result, gf, _ = _get_team_result(row, team)
        results.append(result)
        scored.append(gf > 0)

    unbeaten = 0
    for r in reversed(results):
        if r in ('W', 'D'):
            unbeaten += 1
        else:
            break

    winning = _streak(results, 'W')

    scoring = 0
    for s in reversed(scored):
        if s:
            scoring += 1
        else:
            break

    return {
        f'{prefix}_unbeaten': unbeaten,
        f'{prefix}_winning': winning,
        f'{prefix}_scoring': scoring,
    }


# ---------------------------------------------------------------------------
# GROUP 14: CONSISTENCY (Std dev over last 10)
# ---------------------------------------------------------------------------

def compute_consistency(df, team, before_idx, window=10):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    prefix = 'consist'
    keys = [f'{prefix}_goals_std', f'{prefix}_conceded_std',
            f'{prefix}_shots_std', f'{prefix}_sot_std',
            f'{prefix}_corners_std', f'{prefix}_gd_std']
    if len(all_matches) < 3:
        return {k: 0 for k in keys}

    goals_l, conceded_l, shots_l, sot_l, corners_l, gd_l = [], [], [], [], [], []
    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        goals_l.append(stats['goals_for'])
        conceded_l.append(stats['goals_against'])
        shots_l.append(stats['shots'])
        sot_l.append(stats['sot'])
        corners_l.append(stats['corners'])
        gd_l.append(stats['goals_for'] - stats['goals_against'])

    return {
        f'{prefix}_goals_std': _safe_std(goals_l),
        f'{prefix}_conceded_std': _safe_std(conceded_l),
        f'{prefix}_shots_std': _safe_std(shots_l),
        f'{prefix}_sot_std': _safe_std(sot_l),
        f'{prefix}_corners_std': _safe_std(corners_l),
        f'{prefix}_gd_std': _safe_std(gd_l),
    }


# ---------------------------------------------------------------------------
# GROUP 15: FIRST HALF PERFORMANCE
# ---------------------------------------------------------------------------

def compute_first_half_performance(df, team, before_idx, window=10):
    hist = df.iloc[:before_idx]
    home_matches = hist[hist['HomeTeam'] == team].tail(window)
    away_matches = hist[hist['AwayTeam'] == team].tail(window)
    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').tail(window)

    prefix = 'ht'
    keys = [f'{prefix}_goals_scored_avg', f'{prefix}_goals_conceded_avg',
            f'{prefix}_win_rate', f'{prefix}_draw_rate']
    if len(all_matches) == 0:
        return {k: 0 for k in keys}

    ht_goals_for, ht_goals_against = 0, 0
    ht_wins, ht_draws = 0, 0

    for _, row in all_matches.iterrows():
        stats = _get_team_stats(row, team)
        ht_gf = stats['ht_goals_for']
        ht_ga = stats['ht_goals_against']

        if pd.isna(ht_gf):
            ht_gf = 0
        if pd.isna(ht_ga):
            ht_ga = 0

        ht_goals_for += ht_gf
        ht_goals_against += ht_ga

        htr = row.get('HTR', None)
        team_is_home = (row['HomeTeam'] == team)
        if htr == 'H' and team_is_home:
            ht_wins += 1
        elif htr == 'A' and not team_is_home:
            ht_wins += 1
        elif htr == 'D':
            ht_draws += 1

    n = len(all_matches)
    return {
        f'{prefix}_goals_scored_avg': ht_goals_for / n,
        f'{prefix}_goals_conceded_avg': ht_goals_against / n,
        f'{prefix}_win_rate': ht_wins / n,
        f'{prefix}_draw_rate': ht_draws / n,
    }


# ---------------------------------------------------------------------------
# ADDITIONAL STANDALONE FEATURES
# ---------------------------------------------------------------------------

def compute_win_percentage(df, team, before_idx, window=30):
    all_matches = get_team_all_sorted(df, team, before_idx).tail(window)
    if len(all_matches) == 0:
        return {'win_pct': 0, 'ppm': 0}

    wins, draws = 0, 0
    for _, row in all_matches.iterrows():
        result, _, _ = _get_team_result(row, team)
        if result == 'W':
            wins += 1
        elif result == 'D':
            draws += 1

    n = len(all_matches)
    return {
        'win_pct': wins / n,
        'ppm': (wins * 3 + draws) / n,
    }


def compute_clean_sheet_rate(df, team, venue, before_idx, window=10):
    hist = df.iloc[:before_idx]
    if venue == 'home':
        matches = hist[hist['HomeTeam'] == team].tail(window)
        goal_against_col = 'FTAG'
    else:
        matches = hist[hist['AwayTeam'] == team].tail(window)
        goal_against_col = 'FTHG'

    if len(matches) == 0:
        return {'cs_rate': 0}

    cs = (matches[goal_against_col] == 0).sum()
    return {'cs_rate': cs / len(matches)}


def compute_all_features(df, idx, elo_df=None):
    row = df.iloc[idx]
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    features = {}

    # Multi-window form (3, 5, 10) for both teams
    for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
        for window in [3, 5, 10]:
            form = compute_team_form_for_team(df, team, idx, window=window)
            for k, v in form.items():
                features[f'{prefix}_{k}'] = v

    # Home/Away venue performance (last 10)
    home_perf = compute_venue_performance(df, home_team, 'home', idx)
    for k, v in home_perf.items():
        features[f'home_team_{k}'] = v

    away_perf = compute_venue_performance(df, away_team, 'away', idx)
    for k, v in away_perf.items():
        features[f'away_team_{k}'] = v

    # Head-to-head (expanded)
    h2h = compute_head_to_head(df, home_team, away_team, idx)
    for k, v in h2h.items():
        features[k] = v

    # Per-team features (attacking, defensive, discipline, momentum, etc.)
    for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
        atk = compute_attacking_strength(df, team, idx)
        for k, v in atk.items():
            features[f'{prefix}_{k}'] = v

        defn = compute_defensive_strength(df, team, idx)
        for k, v in defn.items():
            features[f'{prefix}_{k}'] = v

        disc = compute_discipline(df, team, idx)
        for k, v in disc.items():
            features[f'{prefix}_{k}'] = v

        mom = compute_momentum(df, team, idx)
        for k, v in mom.items():
            features[f'{prefix}_{k}'] = v

        eff = compute_goal_efficiency(df, team, idx)
        for k, v in eff.items():
            features[f'{prefix}_{k}'] = v

        dom = compute_match_dominance(df, team, idx)
        for k, v in dom.items():
            features[f'{prefix}_{k}'] = v

        trend = compute_recent_trend(df, team, idx)
        for k, v in trend.items():
            features[f'{prefix}_{k}'] = v

        consist = compute_consistency(df, team, idx)
        for k, v in consist.items():
            features[f'{prefix}_{k}'] = v

        ht_perf = compute_first_half_performance(df, team, idx)
        for k, v in ht_perf.items():
            features[f'{prefix}_{k}'] = v

    # Fixture difficulty (home team at home, away team away)
    fd_home = compute_fixture_difficulty(df, home_team, 'home', idx)
    for k, v in fd_home.items():
        features[f'home_team_{k}'] = v

    fd_away = compute_fixture_difficulty(df, away_team, 'away', idx)
    for k, v in fd_away.items():
        features[f'away_team_{k}'] = v

    # Venue momentum
    venue_mom_home = compute_venue_momentum(df, home_team, 'home', idx)
    for k, v in venue_mom_home.items():
        features[f'home_team_{k}'] = v

    venue_mom_away = compute_venue_momentum(df, away_team, 'away', idx)
    for k, v in venue_mom_away.items():
        features[f'away_team_{k}'] = v

    # Win percentage and points per match
    wp_home = compute_win_percentage(df, home_team, idx)
    for k, v in wp_home.items():
        features[f'home_team_{k}'] = v

    wp_away = compute_win_percentage(df, away_team, idx)
    for k, v in wp_away.items():
        features[f'away_team_{k}'] = v

    # Clean sheet rate
    cs_home = compute_clean_sheet_rate(df, home_team, 'home', idx)
    for k, v in cs_home.items():
        features[f'home_team_{k}'] = v

    cs_away = compute_clean_sheet_rate(df, away_team, 'away', idx)
    for k, v in cs_away.items():
        features[f'away_team_{k}'] = v

    # Elo Rating
    if elo_df is not None:
        features['home_elo'] = elo_df.iloc[idx]['_home_elo']
        features['away_elo'] = elo_df.iloc[idx]['_away_elo']
        features['elo_diff'] = features['home_elo'] - features['away_elo']
    else:
        features['home_elo'] = 1500
        features['away_elo'] = 1500
        features['elo_diff'] = 0

    # xG Proxy (SOT-based approximation)
    home_sot_rate = features.get('home_form_avg_sot_10', 0)
    away_sot_rate = features.get('away_form_avg_sot_10', 0)
    features['home_xg_proxy'] = home_sot_rate * 0.1
    features['away_xg_proxy'] = away_sot_rate * 0.1
    features['xg_proxy_diff'] = features['home_xg_proxy'] - features['away_xg_proxy']

    # Difference features (home minus away for multi-window)
    for window in [3, 5, 10]:
        for metric in ['form_points', 'form_goals_scored', 'form_avg_shots', 'form_avg_sot', 'form_goal_diff']:
            h_key = f'home_{metric}_{window}'
            a_key = f'away_{metric}_{window}'
            features[f'diff_{metric}_{window}'] = features.get(h_key, 0) - features.get(a_key, 0)

    # Home advantage index
    home_pts_10 = features.get('home_form_points_10', 0)
    away_pts_10 = features.get('away_form_points_10', 0)
    features['home_advantage_index'] = home_pts_10 / max(away_pts_10, 1)

    # Metadata
    features['home_team'] = home_team
    features['away_team'] = away_team
    features['date'] = row['Date']
    features['ftr'] = row['FTR']
    features['ftr_encoded'] = row['FTR_encoded']
    features['season'] = row.get('_season', '')

    return features


def run_feature_engineering():
    print('=' * 60)
    print('FEATURE ENGINEERING (v2 — ~125 features)')
    print('=' * 60)

    print('\nLoading cleaned dataset...')
    df = load_cleaned_data()
    print(f'  {len(df)} matches loaded')

    # Add season column for splitting (based on month: Aug+ = new season)
    seasons = []
    for _, row in df.iterrows():
        yr = row['Date'].year
        mo = row['Date'].month
        if mo >= 8:
            seasons.append(f'{yr}-{yr+1}')
        else:
            seasons.append(f'{yr-1}-{yr}')
    df['_season'] = seasons

    df['FTR_encoded'] = df['FTR'].map(FTR_MAP)

    # Precompute Elo ratings for all matches
    print('\n  Computing Elo ratings...')
    elo_df = compute_elo_ratings(df)
    print('  Elo ratings computed.')

    min_matches = 15
    supported_indices = [
        i for i in range(min_matches, len(df))
        if df.iloc[i]['HomeTeam'] in SUPPORTED_CLUBS or df.iloc[i]['AwayTeam'] in SUPPORTED_CLUBS
    ]
    print(f'\n  Skipping first {min_matches} matches (insufficient history)')
    print(f'  {len(supported_indices)} of {len(df) - min_matches} matches involve supported clubs')

    all_features = []
    start_time = time.time()

    for count, i in enumerate(supported_indices):
        if count % 100 == 0:
            elapsed = time.time() - start_time
            pct = count / len(supported_indices) * 100
            print(f'  Progress: {pct:.0f}% ({count}/{len(supported_indices)}) - {elapsed:.1f}s')

        feats = compute_all_features(df, i, elo_df=elo_df)
        all_features.append(feats)

    elapsed = time.time() - start_time
    print(f'\n  Feature engineering complete: {elapsed:.1f}s')

    features_df = pd.DataFrame(all_features)
    print(f'  Output shape (supported clubs only): {features_df.shape}')

    # Convert numeric columns
    meta_cols = ['home_team', 'away_team', 'date', 'ftr', 'ftr_encoded', 'season']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    for col in feature_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
    print(f'  Features: {features_df.shape[1] - len(meta_cols)} (excluding metadata)')

    os.makedirs(os.path.join(RESULTS_DIR, 'engineered_features'), exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'engineered_features', 'engineered_features.csv')
    features_df.to_csv(output_path, index=False)
    print(f'\n  Saved: {output_path}')

    feature_cols_final = [c for c in features_df.columns if c not in meta_cols]
    print(f'\n  Feature columns ({len(feature_cols_final)}):')
    for col in sorted(feature_cols_final):
        print(f'    - {col}')

    return features_df


if __name__ == '__main__':
    run_feature_engineering()
