import pandas as pd
import numpy as np


RETAINED_COLUMNS = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
]

NUMERIC_COLUMNS = [
    'FTHG', 'FTAG', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR',
]

FTR_MAP = {'H': 0, 'D': 1, 'A': 2}

TEAM_NAME_MAP = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Spurs': 'Tottenham',
    'Nott\'m Forest': 'Nottingham Forest',
    'Wolves': 'Wolverhampton',
}

METADATA_COLUMNS = ['_season', '_source_file']


def remove_betting_odds(df):
    keep = RETAINED_COLUMNS + METADATA_COLUMNS
    cols_to_drop = [c for c in df.columns if c not in keep]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f'  Dropped {len(cols_to_drop)} betting/extra columns')
    print(f'  Retained {len([c for c in df.columns if c not in METADATA_COLUMNS])} core columns')
    return df


def convert_dates(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    nat_count = df['Date'].isna().sum()
    if nat_count > 0:
        print(f'  Warning: {nat_count} dates could not be parsed, dropping them')
        df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.normalize()
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f'  Removed {removed} duplicate matches')
    else:
        print('  No duplicates found')
    return df


def validate_numeric_columns(df):
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def handle_missing_values(df):
    df['HomeTeam'] = df['HomeTeam'].fillna('Unknown')
    df['AwayTeam'] = df['AwayTeam'].fillna('Unknown')

    df['FTR'] = df['FTR'].fillna('D')
    df['HTR'] = df['HTR'].fillna('D')

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f'  Remaining missing values: {missing_count} (filling with 0)')
        df = df.fillna(0)
    else:
        print('  No missing values remaining')

    return df


def map_target_variable(df):
    df['FTR_encoded'] = df['FTR'].map(FTR_MAP)

    unmapped = df['FTR_encoded'].isna().sum()
    if unmapped > 0:
        print(f'  Warning: {unmapped} unmapped FTR values, defaulting to Draw (1)')
        df['FTR_encoded'] = df['FTR_encoded'].fillna(1).astype(int)

    df['FTR_encoded'] = df['FTR_encoded'].astype(int)
    return df


def validate_team_names(df):
    supported = ['Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United']
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    found = [t for t in supported if t in all_teams]
    missing = [t for t in supported if t not in all_teams]

    print(f'  Supported clubs found: {found}')
    if missing:
        print(f'  Missing clubs: {missing}')

    return df


def standardize_team_names(df):
    before_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    df['HomeTeam'] = df['HomeTeam'].replace(TEAM_NAME_MAP)
    df['AwayTeam'] = df['AwayTeam'].replace(TEAM_NAME_MAP)
    after_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    mapped = before_teams - after_teams
    if mapped:
        print(f'  Standardized team names: {mapped}')
    return df


def clean_dataset(df):
    print('\n--- Data Cleaning Pipeline ---')
    print(f'Input: {len(df)} rows, {len(df.columns)} columns')

    print('\n[1/8] Converting dates...')
    df = convert_dates(df)

    print('\n[2/8] Removing betting odds columns...')
    df = remove_betting_odds(df)

    print('\n[3/8] Removing duplicates...')
    df = remove_duplicates(df)

    print('\n[4/8] Validating numeric columns...')
    df = validate_numeric_columns(df)

    print('\n[5/8] Handling missing values...')
    df = handle_missing_values(df)

    print('\n[6/8] Mapping target variable (FTR)...')
    df = map_target_variable(df)

    print('\n[7/8] Sorting chronologically...')
    df = df.sort_values('Date').reset_index(drop=True)

    print('\n[8/8] Standardizing team names...')
    df = standardize_team_names(df)

    print('\n  Validating team names...')
    validate_team_names(df)

    core_cols = [c for c in RETAINED_COLUMNS if c in df.columns]
    df = df[core_cols + ['FTR_encoded'] + [c for c in METADATA_COLUMNS if c in df.columns]]

    print(f'\nOutput: {len(df)} rows, {len(df.columns)} columns')
    print(f'Date range: {df["Date"].min().date()} to {df["Date"].max().date()}')

    return df
