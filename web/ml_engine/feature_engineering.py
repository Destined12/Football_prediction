import os
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings


SUPPORTED_CLUBS = [
    'Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United'
]

RETAINED_COLUMNS = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
]

TEAM_NAME_MAP = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Wolves': 'Wolverhampton',
    "Nott'm Forest": 'Nottingham Forest',
    "Spurs": 'Tottenham',
}


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


def _get_team_all_sorted(df, team):
    hist = df
    home = hist[hist['HomeTeam'] == team].copy()
    away = hist[hist['AwayTeam'] == team].copy()
    return pd.concat([home, away]).sort_values('Date')


def _compute_elo_ratings(df):
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
    return elo, home_elos, away_elos


class FeatureEngineering:

    def __init__(self):
        self.dataset = self._load_dataset()
        self._elo_cache = None

    def _load_dataset(self):
        csv_path = settings.DATA_DIR / 'processed' / 'cleaned_dataset.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df['HomeTeam'] = df['HomeTeam'].replace(TEAM_NAME_MAP)
            df['AwayTeam'] = df['AwayTeam'].replace(TEAM_NAME_MAP)
            df = df.sort_values('Date').reset_index(drop=True)
            return df
        return None

    def _get_elo(self):
        if self._elo_cache is None:
            self._elo_cache, _, _ = _compute_elo_ratings(self.dataset)
        return self._elo_cache

    def _compute_team_form(self, team, window=5):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {f'form_{k}_{window}': 0 for k in [
                'wins', 'draws', 'losses', 'points', 'goals_scored',
                'goals_conceded', 'goal_diff', 'avg_goals', 'avg_shots', 'avg_sot',
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

    def _compute_venue_performance(self, team, venue, window=10):
        if venue == 'home':
            matches = self.dataset[self.dataset['HomeTeam'] == team].tail(window)
            gf_col, ga_col, shots_col, sot_col = 'FTHG', 'FTAG', 'HS', 'HST'
            win_val = 'H'
        else:
            matches = self.dataset[self.dataset['AwayTeam'] == team].tail(window)
            gf_col, ga_col, shots_col, sot_col = 'FTAG', 'FTHG', 'AS', 'AST'
            win_val = 'A'

        prefix = f'{venue}_perf'
        if len(matches) == 0:
            return {k: 0 for k in [
                f'{prefix}_win_rate', f'{prefix}_avg_goals', f'{prefix}_avg_conceded',
                f'{prefix}_avg_shots', f'{prefix}_avg_sot',
            ]}

        wins = (matches['FTR'] == win_val).sum()
        n = len(matches)
        return {
            f'{prefix}_win_rate': wins / n,
            f'{prefix}_avg_goals': matches[gf_col].mean(),
            f'{prefix}_avg_conceded': matches[ga_col].mean(),
            f'{prefix}_avg_shots': matches[shots_col].mean(),
            f'{prefix}_avg_sot': matches[sot_col].mean(),
        }

    def _compute_h2h(self, home_team, away_team, window=10):
        h2h = self.dataset[
            ((self.dataset['HomeTeam'] == home_team) & (self.dataset['AwayTeam'] == away_team)) |
            ((self.dataset['HomeTeam'] == away_team) & (self.dataset['AwayTeam'] == home_team))
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
                if row['FTR'] == 'H':
                    hw += 1
                elif row['FTR'] == 'D':
                    dr += 1
                else:
                    aw += 1
            else:
                hg += row['FTAG']
                ag += row['FTHG']
                if row['FTR'] == 'A':
                    hw += 1
                elif row['FTR'] == 'D':
                    dr += 1
                else:
                    aw += 1
            total_shots += row['HS'] + row['AS']
            total_corners += row['HC'] + row['AC']

        n = len(h2h)
        return {
            'h2h_home_wins': hw, 'h2h_away_wins': aw, 'h2h_draws': dr,
            'h2h_home_goals': hg, 'h2h_away_goals': ag,
            'h2h_goal_diff': hg - ag,
            'h2h_avg_total_goals': (hg + ag) / n,
            'h2h_home_scoring_rate': hg / n, 'h2h_away_scoring_rate': ag / n,
            'h2h_avg_shots': total_shots / n, 'h2h_avg_corners': total_corners / n,
            'h2h_home_dominance': (hw - aw) / n,
        }

    def _compute_team_atk(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {k: 0 for k in ['atk_avg_goals', 'atk_avg_shots', 'atk_avg_sot', 'atk_avg_corners']}
        goals, shots, sot, corners = 0, 0, 0, 0
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            goals += stats['goals_for']
            shots += stats['shots']
            sot += stats['sot']
            corners += stats['corners']
        n = len(all_matches)
        return {'atk_avg_goals': goals / n, 'atk_avg_shots': shots / n, 'atk_avg_sot': sot / n, 'atk_avg_corners': corners / n}

    def _compute_team_def(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {k: 0 for k in ['def_avg_conceded', 'def_clean_sheet_rate', 'def_shots_faced', 'def_avg_fouls', 'def_avg_yellow', 'def_avg_red']}
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
        return {'def_avg_conceded': gc / n, 'def_clean_sheet_rate': cs / n, 'def_shots_faced': sf / n, 'def_avg_fouls': fouls / n, 'def_avg_yellow': yellows / n, 'def_avg_red': reds / n}

    def _compute_team_disc(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {'disc_fouls': 0, 'disc_yellow': 0, 'disc_red': 0}
        fouls, yellows, reds = 0, 0, 0
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            fouls += stats['fouls']
            yellows += stats['yellows']
            reds += stats['reds']
        n = len(all_matches)
        return {'disc_fouls': fouls / n, 'disc_yellow': yellows / n, 'disc_red': reds / n}

    def _compute_momentum(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {'mom_win_streak': 0, 'mom_loss_streak': 0, 'mom_unbeaten_streak': 0}
        results = []
        for _, row in all_matches.iterrows():
            result, _, _ = _get_team_result(row, team)
            results.append(result)
        return {'mom_win_streak': _streak(results, 'W'), 'mom_loss_streak': _streak(results, 'L'), 'mom_unbeaten_streak': _unbeaten_streak(results)}

    def _compute_efficiency(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {k: 0 for k in ['eff_goal_conversion', 'eff_shot_accuracy', 'eff_goals_per_sot', 'eff_sot_conversion']}
        goals, shots, sot = 0, 0, 0
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            goals += stats['goals_for']
            shots += stats['shots']
            sot += stats['sot']
        return {'eff_goal_conversion': goals / shots if shots > 0 else 0, 'eff_shot_accuracy': sot / shots if shots > 0 else 0, 'eff_goals_per_sot': goals / sot if sot > 0 else 0, 'eff_sot_conversion': sot / shots if shots > 0 else 0}

    def _compute_dominance(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) == 0:
            return {k: 0 for k in ['dom_shots_diff', 'dom_sot_diff', 'dom_corners_diff', 'dom_goals_diff']}
        shot_d, sot_d, corner_d, goal_d = [], [], [], []
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            shot_d.append(stats['shots'] - stats['shots_against'])
            sot_d.append(stats['sot'] - stats['sot_against'])
            corner_d.append(stats['corners'] - stats['corners_against'])
            goal_d.append(stats['goals_for'] - stats['goals_against'])
        return {'dom_shots_diff': _safe_mean(shot_d), 'dom_sot_diff': _safe_mean(sot_d), 'dom_corners_diff': _safe_mean(corner_d), 'dom_goals_diff': _safe_mean(goal_d)}

    def _compute_trend(self, team, window=5):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) < 3:
            return {k: 0 for k in ['trend_goals', 'trend_conceded', 'trend_shots', 'trend_sot']}
        goals_l, conceded_l, shots_l, sot_l = [], [], [], []
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            goals_l.append(stats['goals_for'])
            conceded_l.append(stats['goals_against'])
            shots_l.append(stats['shots'])
            sot_l.append(stats['sot'])
        return {'trend_goals': _linear_trend(goals_l, window), 'trend_conceded': _linear_trend(conceded_l, window), 'trend_shots': _linear_trend(shots_l, window), 'trend_sot': _linear_trend(sot_l, window)}

    def _compute_consistency(self, team, window=10):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
        if len(all_matches) < 3:
            return {k: 0 for k in ['consist_goals_std', 'consist_conceded_std', 'consist_shots_std', 'consist_sot_std', 'consist_corners_std', 'consist_gd_std']}
        goals_l, conceded_l, shots_l, sot_l, corners_l, gd_l = [], [], [], [], [], []
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            goals_l.append(stats['goals_for'])
            conceded_l.append(stats['goals_against'])
            shots_l.append(stats['shots'])
            sot_l.append(stats['sot'])
            corners_l.append(stats['corners'])
            gd_l.append(stats['goals_for'] - stats['goals_against'])
        return {'consist_goals_std': _safe_std(goals_l), 'consist_conceded_std': _safe_std(conceded_l), 'consist_shots_std': _safe_std(shots_l), 'consist_sot_std': _safe_std(sot_l), 'consist_corners_std': _safe_std(corners_l), 'consist_gd_std': _safe_std(gd_l)}

    def _compute_ht_performance(self, team, window=10):
        home_matches = self.dataset[self.dataset['HomeTeam'] == team].tail(window)
        away_matches = self.dataset[self.dataset['AwayTeam'] == team].tail(window)
        all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').tail(window)
        if len(all_matches) == 0:
            return {k: 0 for k in ['ht_goals_scored_avg', 'ht_goals_conceded_avg', 'ht_win_rate', 'ht_draw_rate']}
        ht_gf, ht_ga, ht_wins, ht_draws = 0, 0, 0, 0
        for _, row in all_matches.iterrows():
            stats = _get_team_stats(row, team)
            ht_gf += stats['ht_goals_for'] if pd.notna(stats['ht_goals_for']) else 0
            ht_ga += stats['ht_goals_against'] if pd.notna(stats['ht_goals_against']) else 0
            htr = row.get('HTR', None)
            team_is_home = (row['HomeTeam'] == team)
            if htr == 'H' and team_is_home:
                ht_wins += 1
            elif htr == 'A' and not team_is_home:
                ht_wins += 1
            elif htr == 'D':
                ht_draws += 1
        n = len(all_matches)
        return {'ht_goals_scored_avg': ht_gf / n, 'ht_goals_conceded_avg': ht_ga / n, 'ht_win_rate': ht_wins / n, 'ht_draw_rate': ht_draws / n}

    def _compute_fixture_difficulty(self, team, venue, window=10):
        if venue == 'home':
            matches = self.dataset[self.dataset['HomeTeam'] == team].tail(window)
            opp_col = 'AwayTeam'
        else:
            matches = self.dataset[self.dataset['AwayTeam'] == team].tail(window)
            opp_col = 'HomeTeam'
        prefix = f'fd_{venue}'
        if len(matches) == 0:
            return {k: 0 for k in [f'{prefix}_opp_win_rate', f'{prefix}_opp_avg_gd', f'{prefix}_opp_avg_points']}
        opp_wr, opp_gd, opp_pts = [], [], []
        for _, row in matches.iterrows():
            opp = row[opp_col]
            opp_home = self.dataset[self.dataset['HomeTeam'] == opp]
            opp_away = self.dataset[self.dataset['AwayTeam'] == opp]
            opp_all = pd.concat([opp_home, opp_away])
            if len(opp_all) < 3:
                opp_wr.append(0.33)
                opp_gd.append(0)
                opp_pts.append(1.0)
                continue
            opp_w = ((opp_all['HomeTeam'] == opp) & (opp_all['FTR'] == 'H')).sum() + ((opp_all['AwayTeam'] == opp) & (opp_all['FTR'] == 'A')).sum()
            opp_d = (opp_all['FTR'] == 'D').sum()
            opp_n = len(opp_all)
            opp_gs, opp_ga = 0, 0
            for _, orow in opp_all.iterrows():
                if orow['HomeTeam'] == opp:
                    opp_gs += orow['FTHG']
                    opp_ga += orow['FTAG']
                else:
                    opp_gs += orow['FTAG']
                    opp_ga += orow['FTHG']
            opp_wr.append(opp_w / opp_n)
            opp_gd.append((opp_gs - opp_ga) / opp_n)
            opp_pts.append((opp_w * 3 + opp_d) / opp_n)
        return {f'{prefix}_opp_win_rate': _safe_mean(opp_wr), f'{prefix}_opp_avg_gd': _safe_mean(opp_gd), f'{prefix}_opp_avg_points': _safe_mean(opp_pts)}

    def _compute_venue_momentum(self, team, venue, window=20):
        if venue == 'home':
            matches = self.dataset[self.dataset['HomeTeam'] == team].tail(window)
        else:
            matches = self.dataset[self.dataset['AwayTeam'] == team].tail(window)
        prefix = f'venue_mom_{venue}'
        if len(matches) == 0:
            return {k: 0 for k in [f'{prefix}_unbeaten', f'{prefix}_winning', f'{prefix}_scoring']}
        results, scored = [], []
        for _, row in matches.iterrows():
            result, gf, _ = _get_team_result(row, team)
            results.append(result)
            scored.append(gf > 0)
        unbeaten = _unbeaten_streak(results)
        winning = _streak(results, 'W')
        scoring = 0
        for s in reversed(scored):
            if s:
                scoring += 1
            else:
                break
        return {f'{prefix}_unbeaten': unbeaten, f'{prefix}_winning': winning, f'{prefix}_scoring': scoring}

    def _compute_win_pct(self, team, window=30):
        all_matches = _get_team_all_sorted(self.dataset, team).tail(window)
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
        return {'win_pct': wins / n, 'ppm': (wins * 3 + draws) / n}

    def _compute_cs_rate(self, team, venue, window=10):
        if venue == 'home':
            matches = self.dataset[self.dataset['HomeTeam'] == team].tail(window)
            ga_col = 'FTAG'
        else:
            matches = self.dataset[self.dataset['AwayTeam'] == team].tail(window)
            ga_col = 'FTHG'
        if len(matches) == 0:
            return {'cs_rate': 0}
        cs = (matches[ga_col] == 0).sum()
        return {'cs_rate': cs / len(matches)}

    def compute_match_features(self, home_team, away_team):
        if self.dataset is None or len(self.dataset) == 0:
            return None

        features = {}

        for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
            for window in [3, 5, 10]:
                form = self._compute_team_form(team, window)
                for k, v in form.items():
                    features[f'{prefix}_{k}'] = v

        home_perf = self._compute_venue_performance(home_team, 'home')
        for k, v in home_perf.items():
            features[f'home_team_{k}'] = v

        away_perf = self._compute_venue_performance(away_team, 'away')
        for k, v in away_perf.items():
            features[f'away_team_{k}'] = v

        h2h = self._compute_h2h(home_team, away_team)
        for k, v in h2h.items():
            features[k] = v

        for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
            for comp, fn in [
                ('atk', self._compute_team_atk), ('def', self._compute_team_def),
                ('disc', self._compute_team_disc), ('mom', self._compute_momentum),
                ('eff', self._compute_efficiency), ('dom', self._compute_dominance),
                ('trend', self._compute_trend), ('consist', self._compute_consistency),
                ('ht', self._compute_ht_performance),
            ]:
                result = fn(team)
                for k, v in result.items():
                    features[f'{prefix}_{k}'] = v

        fd_home = self._compute_fixture_difficulty(home_team, 'home')
        for k, v in fd_home.items():
            features[f'home_team_{k}'] = v

        fd_away = self._compute_fixture_difficulty(away_team, 'away')
        for k, v in fd_away.items():
            features[f'away_team_{k}'] = v

        vm_home = self._compute_venue_momentum(home_team, 'home')
        for k, v in vm_home.items():
            features[f'home_team_{k}'] = v

        vm_away = self._compute_venue_momentum(away_team, 'away')
        for k, v in vm_away.items():
            features[f'away_team_{k}'] = v

        wp_home = self._compute_win_pct(home_team)
        for k, v in wp_home.items():
            features[f'home_team_{k}'] = v

        wp_away = self._compute_win_pct(away_team)
        for k, v in wp_away.items():
            features[f'away_team_{k}'] = v

        cs_home = self._compute_cs_rate(home_team, 'home')
        for k, v in cs_home.items():
            features[f'home_team_{k}'] = v

        cs_away = self._compute_cs_rate(away_team, 'away')
        for k, v in cs_away.items():
            features[f'away_team_{k}'] = v

        elo = self._get_elo()
        features['home_elo'] = elo.get(home_team, 1500)
        features['away_elo'] = elo.get(away_team, 1500)
        features['elo_diff'] = features['home_elo'] - features['away_elo']

        home_sot_rate = features.get('home_form_avg_sot_10', 0)
        away_sot_rate = features.get('away_form_avg_sot_10', 0)
        features['home_xg_proxy'] = home_sot_rate * 0.1
        features['away_xg_proxy'] = away_sot_rate * 0.1
        features['xg_proxy_diff'] = features['home_xg_proxy'] - features['away_xg_proxy']

        for window in [3, 5, 10]:
            for metric in ['form_points', 'form_goals_scored', 'form_avg_shots', 'form_avg_sot', 'form_goal_diff']:
                h_key = f'home_{metric}_{window}'
                a_key = f'away_{metric}_{window}'
                features[f'diff_{metric}_{window}'] = features.get(h_key, 0) - features.get(a_key, 0)

        home_pts_10 = features.get('home_form_points_10', 0)
        away_pts_10 = features.get('away_form_points_10', 0)
        features['home_advantage_index'] = home_pts_10 / max(away_pts_10, 1)

        return pd.DataFrame([features])
