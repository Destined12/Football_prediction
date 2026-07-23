import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, SUPPORTED_CLUBS, filter_supported


def figure_4_8_h2h_heatmap(df):
    n = len(SUPPORTED_CLUBS)
    h2h_wins = pd.DataFrame(0, index=SUPPORTED_CLUBS, columns=SUPPORTED_CLUBS)

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        if ht in SUPPORTED_CLUBS and at in SUPPORTED_CLUBS and ht != at:
            if row['FTR'] == 'H':
                h2h_wins.loc[ht, at] += 1
            elif row['FTR'] == 'A':
                h2h_wins.loc[at, ht] += 1

    for club in SUPPORTED_CLUBS:
        h2h_wins.loc[club, club] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(n, dtype=bool)
    sns_heatmap = __import__('seaborn')
    sns_heatmap.heatmap(
        h2h_wins, annot=True, fmt='d', cmap='YlGnBu',
        mask=mask, ax=ax, linewidths=0.5, linecolor='#334155',
        annot_kws={'size': 12, 'fontweight': 'bold'},
        cbar_kws={'label': 'Home Wins'}
    )
    ax.set_title('Head-to-Head Home Wins (Supported Clubs)', fontweight='bold')
    ax.set_xlabel('Away Team')
    ax.set_ylabel('Home Team')
    fig.tight_layout()
    save_fig(fig, 'fig_4_8_h2h_heatmap')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_8_h2h_heatmap(df)
