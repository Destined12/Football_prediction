import os
import sys
import importlib
import time

EXPLORATORY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPLORATORY_DIR)

SCRIPTS = [
    '01_data_overview',
    '02_outcome_distribution',
    '03_goals_analysis',
    '04_team_performance',
    '05_form_analysis',
    '06_h2h_analysis',
    '07_correlation_analysis',
    '09_feature_importance_preliminary',
]


SCRIPTS_NO_DF = ['07_correlation_analysis', '09_feature_importance_preliminary']


def run_all():
    print('=' * 60)
    print('RUNNING ALL EDA SCRIPTS')
    print('=' * 60)

    total_start = time.time()
    from config import load_data, filter_supported
    df = filter_supported(load_data())

    for script_name in SCRIPTS:
        print(f'\n--- {script_name} ---')
        start = time.time()
        try:
            mod = importlib.import_module(script_name)
            funcs = [getattr(mod, attr) for attr in dir(mod)
                     if attr.startswith('figure_')]
            for func in funcs:
                if script_name in SCRIPTS_NO_DF:
                    func()
                else:
                    func(df)
            elapsed = time.time() - start
            print(f'  Completed in {elapsed:.1f}s')
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f'\n{"=" * 60}')
    print(f'ALL EDA SCRIPTS COMPLETE ({total_elapsed:.1f}s)')
    print(f'Figures saved to: results/figures/eda/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    run_all()
