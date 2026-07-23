import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from config import load_data, save_fig, FIGURES_DIR, filter_supported


def figure_4_1_dataset_summary(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    info_data = []
    for col in df.columns:
        info_data.append({
            'Column': col,
            'Dtype': str(df[col].dtype),
            'Non-Null': df[col].notna().sum(),
            'Missing %': round(df[col].isna().mean() * 100, 1),
            'Unique': df[col].nunique(),
        })
    info_df = pd.DataFrame(info_data)
    table = ax.table(
        cellText=info_df.values,
        colLabels=info_df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#10b981')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#1e293b')
        else:
            cell.set_facecolor('#0f172a')
        cell.set_edgecolor('#334155')
    ax.set_title('Dataset Summary', fontweight='bold')
    ax.axis('off')

    ax2 = axes[1]
    seasons = df['_season'].value_counts().sort_index()
    bars = ax2.barh(seasons.index, seasons.values, color='#10b981', edgecolor='#059669')
    for bar, val in zip(bars, seasons.values):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=10, color='#e2e8f0')
    ax2.set_title('Matches per Season', fontweight='bold')
    ax2.set_xlabel('Number of Matches')

    fig.suptitle('Dataset Summary', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_4_1_dataset_summary')


if __name__ == '__main__':
    df = filter_supported(load_data())
    figure_4_1_dataset_summary(df)
