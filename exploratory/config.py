import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DPI = 300
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_dataset.csv')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'eda')

SUPPORTED_CLUBS = [
    'Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United'
]

COLORS = {
    'Arsenal': '#EF0107',
    'Liverpool': '#C8102E',
    'Manchester City': '#6CABDD',
    'Chelsea': '#034694',
    'Manchester United': '#DA291C',
}

sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#cbd5e1',
    'text.color': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#334155',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': DPI,
})


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    return df


def filter_supported(df):
    return df[df['HomeTeam'].isin(SUPPORTED_CLUBS) | df['AwayTeam'].isin(SUPPORTED_CLUBS)].copy()


def save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path_png = os.path.join(FIGURES_DIR, f'{name}.png')
    path_svg = os.path.join(FIGURES_DIR, f'{name}.svg')
    fig.savefig(path_png, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(path_svg, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {name}.png, {name}.svg')
