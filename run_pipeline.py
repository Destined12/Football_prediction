import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web'))

import pandas as pd
from dataset.data_loader import load_all_seasons, sort_chronological, get_season_info
from dataset.data_cleaner import clean_dataset

DATA_RAW = os.path.join(os.path.dirname(__file__), 'data', 'raw')
DATA_PROCESSED = os.path.join(os.path.dirname(__file__), 'data', 'processed')


def run_pipeline(include_validation=False):
    print('=' * 60)
    print('FOOTBALL DATA PIPELINE')
    print('=' * 60)

    print('\n[STEP 1] Loading raw CSV files...')
    df = load_all_seasons(include_validation=include_validation)

    print('\n[STEP 2] Sorting chronologically...')
    df = sort_chronological(df)
    print(f'  After date parsing: {len(df)} rows')

    print('\n[STEP 3] Season breakdown:')
    for info in get_season_info(df):
        print(f'  {info["season"]}: {info["rows"]} matches ({info["date_start"].date()} to {info["date_end"].date()})')

    print('\n[STEP 4] Cleaning dataset...')
    df = clean_dataset(df)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED, 'cleaned_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f'\n[STEP 5] Saved cleaned dataset to: {output_path}')

    print('\n--- Column Summary ---')
    print(f'Columns: {list(df.columns)}')
    print(f'\nFirst 3 rows:')
    print(df.head(3).to_string())

    print('\n--- Value Counts (FTR) ---')
    print(df['FTR'].value_counts().to_string())

    print('\n--- Supported Clubs Filter Check ---')
    supported = ['Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United']
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    matches_with_supported = df[
        (df['HomeTeam'].isin(supported)) | (df['AwayTeam'].isin(supported))
    ]
    print(f'Total teams in dataset: {len(all_teams)}')
    print(f'Supported clubs present: {len([t for t in supported if t in all_teams])}')
    print(f'Matches involving supported clubs: {len(matches_with_supported)}')
    print(f'Total matches: {len(df)}')

    print('\n' + '=' * 60)
    print('PIPELINE COMPLETE')
    print('=' * 60)

    return df


if __name__ == '__main__':
    include_val = '--with-validation' in sys.argv
    run_pipeline(include_validation=include_val)
