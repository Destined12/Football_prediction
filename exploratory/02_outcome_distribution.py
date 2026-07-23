import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, filter_supported


def figure_4_2_outcome_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    counts = df['FTR'].value_counts().sort_index()
    labels = ['Home Win', 'Draw', 'Away Win']
    colors = ['#10b981', '#eab308', '#ef4444']
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='#1e293b', width=0.6)
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, val + 10,
                f'{val}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Match Outcome Distribution', fontweight='bold')
    ax.set_ylabel('Number of Matches')
    ax.set_ylim(0, max(counts.values) * 1.2)

    ax2 = axes[1]
    season_ftr = df.groupby(['_season', 'FTR']).size().unstack(fill_value=0)
    season_ftr_pct = season_ftr.div(season_ftr.sum(axis=1), axis=0) * 100
    season_ftr_pct.plot(kind='bar', stacked=True, ax=ax2,
                        color=['#10b981', '#eab308', '#ef4444'],
                        edgecolor='#1e293b')
    ax2.set_title('Outcome Distribution by Season', fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Season')
    ax2.legend(['Home Win', 'Draw', 'Away Win'], loc='upper right', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)

    fig.suptitle('Match Outcome Distribution', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_4_2_outcome_distribution')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_2_outcome_distribution(df)
