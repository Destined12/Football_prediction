import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, SUPPORTED_CLUBS, COLORS, filter_supported


def compute_rolling_form(df, team, window=5):
    team_home = df[df['HomeTeam'] == team].copy()
    team_away = df[df['AwayTeam'] == team].copy()

    team_home['points'] = team_home['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    team_home['goals_for'] = team_home['FTHG']
    team_home['goals_against'] = team_home['FTAG']

    team_away['points'] = team_away['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    team_away['goals_for'] = team_away['FTAG']
    team_away['goals_against'] = team_away['FTHG']

    matches = pd.concat([team_home[['Date', 'points', 'goals_for', 'goals_against']],
                         team_away[['Date', 'points', 'goals_for', 'goals_against']]])
    matches = matches.sort_values('Date').reset_index(drop=True)

    matches['rolling_points'] = matches['points'].rolling(window, min_periods=1).mean() * 3
    matches['rolling_goals'] = matches['goals_for'].rolling(window, min_periods=1).mean()
    return matches


def figure_4_7_team_form(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    ax = axes[0]
    for club in SUPPORTED_CLUBS:
        form = compute_rolling_form(df, club, window=5)
        ax.plot(form['Date'], form['rolling_points'], label=club,
                color=COLORS.get(club, '#94a3b8'), linewidth=1.5)
    ax.set_title('Rolling Points Trend (Last 5 Matches)', fontweight='bold')
    ax.set_ylabel('Average Points per Game (x3)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for club in SUPPORTED_CLUBS:
        form = compute_rolling_form(df, club, window=5)
        ax2.plot(form['Date'], form['rolling_goals'], label=club,
                 color=COLORS.get(club, '#94a3b8'), linewidth=1.5)
    ax2.set_title('Rolling Goals Scored Trend (Last 5 Matches)', fontweight='bold')
    ax2.set_ylabel('Average Goals per Match')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig_4_7_team_form_trend')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_7_team_form(df)
