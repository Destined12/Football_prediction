import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, SUPPORTED_CLUBS, COLORS, filter_supported


def figure_4_4_avg_goals_scored(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    home_goals = df.groupby('HomeTeam')['FTHG'].mean()
    away_goals = df.groupby('AwayTeam')['FTAG'].mean()
    combined = pd.DataFrame({'Home': home_goals, 'Away': away_goals}).fillna(0)
    combined = combined[combined.index.isin(SUPPORTED_CLUBS)]
    combined['Total'] = combined['Home'] + combined['Away']
    combined = combined.sort_values('Total', ascending=True)
    club_colors = [COLORS.get(t, '#94a3b8') for t in combined.index]

    y = range(len(combined))
    ax.barh(y, combined['Home'], color=club_colors, edgecolor='#0f172a', label='Home Goals', height=0.4, alpha=0.9)
    ax.barh([i + 0.4 for i in y], combined['Away'], color=club_colors, edgecolor='#0f172a', label='Away Goals', height=0.4, alpha=0.5)
    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(combined.index)
    ax.set_title('Average Goals Scored by Club', fontweight='bold')
    ax.set_xlabel('Average Goals per Match')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'fig_4_4_avg_goals_scored')


def figure_4_5_avg_goals_conceded(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    home_conceded = df.groupby('HomeTeam')['FTAG'].mean()
    away_conceded = df.groupby('AwayTeam')['FTHG'].mean()
    combined = pd.DataFrame({'Home': home_conceded, 'Away': away_conceded}).fillna(0)
    combined = combined[combined.index.isin(SUPPORTED_CLUBS)]
    combined['Total'] = combined['Home'] + combined['Away']
    combined = combined.sort_values('Total', ascending=True)
    club_colors = [COLORS.get(t, '#94a3b8') for t in combined.index]

    y = range(len(combined))
    ax.barh(y, combined['Home'], color=club_colors, edgecolor='#0f172a', label='Home Conceded', height=0.4, alpha=0.9)
    ax.barh([i + 0.4 for i in y], combined['Away'], color=club_colors, edgecolor='#0f172a', label='Away Conceded', height=0.4, alpha=0.5)
    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(combined.index)
    ax.set_title('Average Goals Conceded by Club', fontweight='bold')
    ax.set_xlabel('Average Goals Conceded per Match')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'fig_4_5_avg_goals_conceded')


def figure_4_6_home_vs_away(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ('FTHG', 'FTAG', 'Goals Scored', '#10b981'),
        ('HST', 'AST', 'Shots on Target', '#3b82f6'),
        ('HS', 'AS', 'Total Shots', '#a855f7'),
    ]

    for ax, (home_col, away_col, title, color) in zip(axes, metrics):
        home_avg = df.groupby('HomeTeam')[home_col].mean().reindex(SUPPORTED_CLUBS).dropna()
        away_avg = df.groupby('AwayTeam')[away_col].mean().reindex(SUPPORTED_CLUBS).dropna()
        teams = sorted(set(home_avg.index) & set(away_avg.index))
        x = range(len(teams))
        width = 0.35
        home_vals = [home_avg[t] for t in teams]
        away_vals = [away_avg[t] for t in teams]
        ax.bar([i - width/2 for i in x], home_vals, width, label='Home', color=color, edgecolor='#0f172a')
        ax.bar([i + width/2 for i in x], away_vals, width, label='Away', color='#64748b', edgecolor='#0f172a')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('Manchester ', 'Man ') for t in teams], rotation=30, fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('Home vs Away Performance', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_4_6_home_vs_away_performance')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_4_avg_goals_scored(df)
    figure_4_5_avg_goals_conceded(df)
    figure_4_6_home_vs_away(df)
