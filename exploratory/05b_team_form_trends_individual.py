import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, SUPPORTED_CLUBS, COLORS, FIGURES_DIR


def compute_rolling_form(df, team, window=5):
    team_home = df[df['HomeTeam'] == team].copy()
    team_away = df[df['AwayTeam'] == team].copy()

    team_home['points'] = team_home['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    team_home['goals_for'] = team_home['FTHG']
    team_home['goals_against'] = team_home['FTAG']
    team_home['result'] = team_home['FTR'].apply(lambda x: 'W' if x == 'H' else ('D' if x == 'D' else 'L'))

    team_away['points'] = team_away['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    team_away['goals_for'] = team_away['FTAG']
    team_away['goals_against'] = team_away['FTHG']
    team_away['result'] = team_away['FTR'].apply(lambda x: 'W' if x == 'A' else ('D' if x == 'D' else 'L'))

    matches = pd.concat([team_home[['Date', 'points', 'goals_for', 'goals_against', 'result']],
                         team_away[['Date', 'points', 'goals_for', 'goals_against', 'result']]])
    matches = matches.sort_values('Date').reset_index(drop=True)

    matches['rolling_points'] = matches['points'].rolling(window, min_periods=1).mean() * 3
    matches['rolling_goals_for'] = matches['goals_for'].rolling(window, min_periods=1).mean()
    matches['rolling_goals_against'] = matches['goals_against'].rolling(window, min_periods=1).mean()
    matches['rolling_gd'] = matches['rolling_goals_for'] - matches['rolling_goals_against']

    matches['match_num'] = range(1, len(matches) + 1)
    return matches


def team_form_trend(team, df):
    form = compute_rolling_form(df, team, window=5)
    color = COLORS.get(team, '#94a3b8')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{team} — Form Trend', fontsize=18, fontweight='bold', y=0.98)

    result_colors = {'W': '#10b981', 'D': '#f59e0b', 'L': '#ef4444'}

    # Subplot 1: Results bar + rolling points
    ax1 = axes[0]
    bar_colors = [result_colors[r] for r in form['result']]
    ax1.bar(form['match_num'], [1] * len(form), color=bar_colors, alpha=0.3, width=1.0)
    ax1.plot(form['match_num'], form['rolling_points'], color=color, linewidth=2.5, label='Rolling PPG (x3)', zorder=5)
    ax1.axhline(y=6.0, color='#475569', linestyle='--', linewidth=0.8, alpha=0.5, label='6 pts (2W baseline)')
    ax1.axhline(y=9.0, color='#475569', linestyle=':', linewidth=0.8, alpha=0.5, label='9 pts (3W baseline)')
    ax1.set_ylabel('Points per Game (x3)', fontsize=11)
    ax1.set_title('Rolling Points Trend (5-match window)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 9.5)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.2)

    # Add W/D/L legend patches
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.3, label=l) for l, c in result_colors.items()]
    ax1_leg = ax1.legend(handles=[ax1.get_legend_handles_labels()[0][0], ax1.get_legend_handles_labels()[0][1]] + legend_patches,
                         fontsize=8, loc='lower right', ncol=2)
    ax1.add_artist(ax1_leg)

    # Subplot 2: Goals scored vs conceded
    ax2 = axes[1]
    ax2.fill_between(form['match_num'], form['rolling_goals_for'], alpha=0.2, color='#10b981')
    ax2.plot(form['match_num'], form['rolling_goals_for'], color='#10b981', linewidth=2, label='Goals Scored (avg)')
    ax2.fill_between(form['match_num'], form['rolling_goals_against'], alpha=0.2, color='#ef4444')
    ax2.plot(form['match_num'], form['rolling_goals_against'], color='#ef4444', linewidth=2, label='Goals Conceded (avg)')
    ax2.set_ylabel('Goals per Match', fontsize=11)
    ax2.set_title('Attacking vs Defensive Output (5-match rolling avg)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Subplot 3: Goal difference trend
    ax3 = axes[2]
    pos_gd = form['rolling_gd'] >= 0
    neg_gd = form['rolling_gd'] < 0
    ax3.bar(form['match_num'][pos_gd], form['rolling_gd'][pos_gd], color='#10b981', alpha=0.7, width=1.0)
    ax3.bar(form['match_num'][neg_gd], form['rolling_gd'][neg_gd], color='#ef4444', alpha=0.7, width=1.0)
    ax3.axhline(y=0, color='#e2e8f0', linewidth=1)
    ax3.set_ylabel('Goal Difference', fontsize=11)
    ax3.set_xlabel('Match Number', fontsize=11)
    ax3.set_title('Goal Difference Trend (5-match rolling avg)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f'fig_form_trend_{team.lower().replace(" ", "_")}')


if __name__ == '__main__':
    df = load_data()
    for club in SUPPORTED_CLUBS:
        print(f'Generating form trend for {club}...')
        team_form_trend(club, df)
    print('Done.')
