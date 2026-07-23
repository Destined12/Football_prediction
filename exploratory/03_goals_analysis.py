import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, filter_supported


def figure_4_3_goals_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Home goals
    ax1 = axes[0]
    bins = range(0, int(df['FTHG'].max()) + 2)
    ax1.hist(df['FTHG'], bins=bins, color='#10b981', edgecolor='#0f172a', alpha=0.85, rwidth=0.8)
    ax1.set_title('Home Goals Distribution', fontweight='bold')
    ax1.set_xlabel('Goals')
    ax1.set_ylabel('Frequency')
    mean_val = df['FTHG'].mean()
    ax1.axvline(mean_val, color='#e2e8f0', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax1.legend()

    # Away goals
    ax2 = axes[1]
    bins = range(0, int(df['FTAG'].max()) + 2)
    ax2.hist(df['FTAG'], bins=bins, color='#ef4444', edgecolor='#0f172a', alpha=0.85, rwidth=0.8)
    ax2.set_title('Away Goals Distribution', fontweight='bold')
    ax2.set_xlabel('Goals')
    ax2.set_ylabel('Frequency')
    mean_val = df['FTAG'].mean()
    ax2.axvline(mean_val, color='#e2e8f0', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax2.legend()

    fig.suptitle('Figure 4.3: Goals Distribution (Home vs Away)', fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_4_3_goals_distribution')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_3_goals_distribution(df)
