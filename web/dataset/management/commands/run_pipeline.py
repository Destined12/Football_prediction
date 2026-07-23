import os
import sys
import pandas as pd
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

sys.path.insert(0, str(settings.BASE_DIR.parent.parent))

from dataset.data_loader import load_all_seasons, sort_chronological, get_season_info
from dataset.data_cleaner import clean_dataset
from dataset.models import DatasetInfo


class Command(BaseCommand):
    help = 'Run the data pipeline: load, clean, and populate dataset metadata'

    def add_arguments(self, parser):
        parser.add_argument(
            '--with-validation',
            action='store_true',
            help='Also load the 2025-2026 validation season',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting data pipeline...'))

        include_val = options['with_validation']

        self.stdout.write('Loading raw CSV files...')
        df = load_all_seasons(include_validation=include_val)

        self.stdout.write('Sorting chronologically...')
        df = sort_chronological(df)

        self.stdout.write('Cleaning dataset...')
        df = clean_dataset(df)

        processed_dir = settings.DATA_DIR / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / 'cleaned_dataset.csv'
        df.to_csv(output_path, index=False)
        self.stdout.write(self.style.SUCCESS(f'Saved: {output_path}'))

        self.stdout.write('Populating DatasetInfo...')
        DatasetInfo.objects.all().delete()
        for info in get_season_info(df):
            DatasetInfo.objects.create(
                filename=info['filename'],
                season=info['season'],
                rows=info['rows'],
                columns=info['columns'],
                date_range_start=info['date_start'].date() if info['date_start'] else None,
                date_range_end=info['date_end'].date() if info['date_end'] else None,
            )

        self.stdout.write(self.style.SUCCESS(
            f'Done! {DatasetInfo.objects.count()} seasons recorded.'
        ))
