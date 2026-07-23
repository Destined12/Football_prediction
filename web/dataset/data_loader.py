import os
import pandas as pd
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'

TRAIN_SEASONS = {
    'E0_2019_2020.csv': '2019-2020',
    'E0_2020_2021.csv': '2020-2021',
    'E0_2021_2022.csv': '2021-2022',
    'E0_2022_2023.csv': '2022-2023',
    'E0_2023_2024.csv': '2023-2024',
    'E0_2024_2025.csv': '2024-2025',
}

VALIDATION_FILE = 'E0_2025_2026_validation.csv'


def load_season(filepath, season_label):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    meta = pd.DataFrame({
        '_season': [season_label] * len(df),
        '_source_file': [os.path.basename(filepath)] * len(df),
    })
    df = pd.concat([df, meta], axis=1)
    return df


def load_all_seasons(include_validation=False):
    frames = []

    for filename, season in TRAIN_SEASONS.items():
        filepath = RAW_DIR / filename
        if filepath.exists():
            df = load_season(filepath, season)
            frames.append(df)
            print(f'  Loaded {season}: {len(df)} matches, {len(df.columns)} columns')

    if include_validation:
        filepath = RAW_DIR / VALIDATION_FILE
        if filepath.exists():
            df = load_season(filepath, '2025-2026')
            frames.append(df)
            print(f'  Loaded 2025-2026 (validation): {len(df)} matches')

    if not frames:
        raise FileNotFoundError('No CSV files found in data/raw/')

    merged = pd.concat(frames, ignore_index=True)
    print(f'\n  Total merged: {len(merged)} matches')
    return merged


def sort_chronological(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def get_season_info(df):
    info = []
    for season in df['_season'].unique():
        season_df = df[df['_season'] == season]
        info.append({
            'season': season,
            'filename': season_df['_source_file'].iloc[0],
            'rows': len(season_df),
            'columns': len(season_df.columns),
            'date_start': season_df['Date'].min(),
            'date_end': season_df['Date'].max(),
        })
    return info
