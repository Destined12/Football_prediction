# Implementation Plan: Football Match Prediction System

## Project Title

**"Development of a Web-Based Football Match Prediction System Using a Stacking Ensemble Machine Learning Model."**

---

## Current State

- **3 empty directories**: `exploratory/`, `model/`, `web/`
- **7 CSV files** (2019–2026 seasons), no code, no config, no virtual environment
- Only **6 seasons** needed (2019–2020 through 2024–2025); the 7th file (`E0.csv` = 2025–2026) is used for validation
- **No Python environment** is set up — no `requirements.txt`, no virtual environment
- **No version control** — no `.gitignore`

---

## Decisions

| Decision | Answer |
|----------|--------|
| Django location | `web/` directory (live project) |
| 2025–2026 CSV | Used for validation only |
| Frontend | Tailwind CSS via `django-tailwind-cli` (no Node.js) + custom premium design |
| Deployment | Render.com (PostgreSQL, WhiteNoise, Gunicorn) |
| EDA format | Standalone Python scripts in `exploratory/` |
| Total phases | 10 phases, ~90+ files |

---

## Project Structure

```
C:\Users\Andrew\Desktop\football data\
│
├── exploratory/                    ← EDA Python scripts
│   ├── 01_data_overview.py
│   ├── 02_outcome_distribution.py
│   ├── 03_goals_analysis.py
│   ├── 04_team_performance.py
│   ├── 05_form_analysis.py
│   ├── 06_h2h_analysis.py
│   ├── 07_correlation_analysis.py
│   ├── 08_feature_correlation.py
│   └── 09_feature_importance_preliminary.py
│
├── model/                          ← ML training scripts
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── utils.py
│
├── web/                            ← LIVE DJANGO PROJECT
│   ├── manage.py
│   ├── requirements.txt
│   ├── Procfile                    ← Render deployment
│   ├── render.yaml                 ← Render config
│   ├── build.sh                    ← Render build script
│   │
│   ├── config/                     ← Django project settings
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── asgi.py
│   │
│   ├── core/                       ← Home, About, Model Info, etc.
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── admin.py
│   │
│   ├── prediction/                 ← Prediction logic + history
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── forms.py
│   │   ├── ml_service.py           ← loads models, runs predictions
│   │   └── admin.py
│   │
│   ├── ml_engine/                  ← Feature engineering + model wrappers
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py
│   │   ├── model_training.py
│   │   ├── evaluation.py
│   │   ├── explainability.py
│   │   └── admin.py
│   │
│   ├── dataset/                    ← Data loading + info
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── data_loader.py
│   │   └── admin.py
│   │
│   ├── reports/                    ← Export functionality
│   │   ├── __init__.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── exporters.py
│   │
│   ├── visualization/              ← Serves EDA/model figures
│   │   ├── __init__.py
│   │   ├── views.py
│   │   └── urls.py
│   │
│   ├── templates/                  ← All HTML templates
│   │   ├── base.html               ← Premium Tailwind base layout
│   │   ├── core/
│   │   │   ├── home.html
│   │   │   ├── about.html
│   │   │   ├── model_info.html
│   │   │   ├── dataset_info.html
│   │   │   └── methodology.html
│   │   ├── prediction/
│   │   │   ├── prediction.html     ← Main prediction UI
│   │   │   └── history.html
│   │   ├── ml_engine/
│   │   │   └── features.html
│   │   └── reports/
│   │       └── reports.html
│   │
│   ├── static/                     ← Tailwind CSS + custom assets
│   │   ├── css/
│   │   │   └── source.css          ← Tailwind input file
│   │   ├── js/
│   │   │   └── prediction.js       ← AJAX prediction + animations
│   │   └── images/
│   │
│   ├── media/
│   │   └── figures/                ← Generated plots served to web
│   │
│   ├── trained_models/             ← Serialized ML models
│   │   ├── catboost_model.cbm
│   │   ├── lightgbm_model.txt
│   │   ├── xgboost_model.json
│   │   ├── logistic_meta_model.pkl
│   │   └── feature_pipeline.pkl
│   │
│   └── db.sqlite3                  ← Local dev DB (PostgreSQL on Render)
│
├── results/                        ← All generated outputs
│   ├── figures/
│   │   ├── eda/                    ← Figures 4.1–4.12
│   │   ├── model/                  ← Figures 4.13–4.19
│   │   └── explainability/         ← Figures 4.20–4.22
│   ├── tables/
│   │   ├── evaluation_metrics.csv
│   │   ├── classification_report.csv
│   │   └── selected_features.csv
│   ├── feature_reports/
│   │   ├── engineered_features.csv
│   │   ├── feature_descriptions.csv
│   │   ├── feature_statistics.csv
│   │   ├── feature_dictionary.xlsx
│   │   └── Feature Engineering Report.pdf
│   └── engineered_features/
│
├── data/
│   ├── raw/                        ← Copy of source CSVs
│   │   ├── E0_2019_2020.csv
│   │   ├── E0_2020_2021.csv
│   │   ├── E0_2021_2022.csv
│   │   ├── E0_2022_2023.csv
│   │   ├── E0_2023_2024.csv
│   │   └── E0_2024_2025.csv
│   └── processed/
│       └── cleaned_dataset.csv
│
├── requirements.txt                ← Root requirements (for Render)
├── .gitignore
└── README.md
```

---

## Phase Breakdown (Execution Order)

### Phase 0: Project Scaffolding

**Location**: `web/` (Django project root)

**Tasks**:

1. **Create Django project skeleton** inside `web/` with `config/` settings module
2. **Create 6 Django apps**: `core`, `prediction`, `ml_engine`, `dataset`, `reports`, `visualization`
3. **Install `django-tailwind-cli`** and configure — no Node.js needed, downloads standalone Tailwind Go binary via pip
4. **Set up WhiteNoise** for compressed static file serving
5. **Configure `settings.py`** for both local dev (SQLite) and Render (PostgreSQL via `dj-database-url`)
6. **Create deployment files**: `requirements.txt`, `Procfile`, `render.yaml`, `build.sh`
7. **Set up Tailwind CSS** source file in `static/css/source.css`

**Django Settings Configuration**:

```python
# config/settings.py

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'tailwind',
    # Local apps
    'core',
    'prediction',
    'ml_engine',
    'dataset',
    'reports',
    'visualization',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # right after SecurityMiddleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Database — local dev uses SQLite, Render uses PostgreSQL
import dj_database_url
DATABASES = {
    'default': dj_database_url.config(
        default='sqlite:///db.sqlite3',
        conn_max_age=600,
    )
}

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Tailwind
TAILWIND_APP_NAME = 'tailwind'
```

**Files Created**:
- `web/config/__init__.py`
- `web/config/settings.py`
- `web/config/urls.py`
- `web/config/wsgi.py`
- `web/config/asgi.py`
- `web/manage.py`
- `web/requirements.txt`
- `web/Procfile`
- `web/render.yaml`
- `web/build.sh`
- `web/static/css/source.css`
- Root `requirements.txt`
- Root `.gitignore`

---

### Phase 1: Data Pipeline (Dataset App)

**Location**: `web/dataset/` and `data/` directories

**Tasks**:

1. **Copy the 6 CSV files** (2019–2020 through 2024–2025) into `data/raw/` with clean naming:
   - `E0 (6).csv` → `E0_2019_2020.csv`
   - `E0 (5).csv` → `E0_2020_2021.csv`
   - `E0 (4).csv` → `E0_2021_2022.csv`
   - `E0 (3).csv` → `E0_2022_2023.csv`
   - `E0 (2).csv` → `E0_2023_2024.csv`
   - `E0 (1).csv` → `E0_2024_2025.csv`

2. **Create `web/dataset/data_loader.py`**:
   - Load all CSVs
   - Merge into single DataFrame
   - Sort chronologically by Date

3. **Create `web/dataset/data_cleaner.py`**:
   - Convert Date column to datetime (handle `DD/MM/YYYY` and `DD/MM/YYYY HH:MM` formats)
   - Remove exact duplicate rows
   - Handle missing values (drop rows with missing FTR, fill numeric NaNs with 0)
   - Validate numeric columns (FTHG, FTAG, HTHG, HTAG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR)

4. **Column filtering** — Drop ALL columns matching betting odds patterns:
   - Drop: `Div`, `Time`, `Referee`
   - Drop all columns containing: `B365`, `BW`, `BF`, `PS`, `Max`, `Avg`, `AH`, `VC`, `IW`, `CL`, `LB`, `BFD`, `BMGM`, `BV`, `BFE`, `P>`, `P<`, any column not in the 21 retained columns

5. **Retain only these columns**:
   ```
   Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
   ```

6. **Map FTR**: `H → 0`, `D → 1`, `A → 2`

7. **Export cleaned dataset** to `data/processed/cleaned_dataset.csv`

8. **Create Django model** (`DatasetInfo`) to store metadata:
   - filename, season, rows, columns, date_loaded

**Key Implementation Detail**:
```python
# Date parsing — flexible across seasons
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
# Some rows may have time component, strip it
df['Date'] = df['Date'].dt.normalize()
```

**Files Created**:
- `web/dataset/__init__.py`
- `web/dataset/models.py`
- `web/dataset/views.py`
- `web/dataset/urls.py`
- `web/dataset/admin.py`
- `web/dataset/data_loader.py`
- `web/dataset/data_cleaner.py`
- `data/raw/*.csv` (6 files)
- `data/processed/cleaned_dataset.csv`

---

### Phase 2: Exploratory Data Analysis (EDA)

**Location**: `exploratory/` folder

**Tasks**:

1. **Create 9 EDA Python scripts** in `exploratory/`:

   | Script | Description | Figure(s) |
   |--------|-------------|-----------|
   | `01_data_overview.py` | Shape, dtypes, missing values, describe, column summary | Figure 4.1 |
   | `02_outcome_distribution.py` | FTR bar charts, home/away/draw ratios per season | Figure 4.2 |
   | `03_goals_analysis.py` | FTHG/FTAG histograms, goal distributions | Figures 4.3, 4.4 |
   | `04_team_performance.py` | Avg goals scored/conceded per club, home vs away | Figures 4.5, 4.6, 4.7 |
   | `05_form_analysis.py` | Rolling form trends over seasons for supported clubs | Figure 4.8 |
   | `06_h2h_analysis.py` | Head-to-head records between supported clubs | Figure 4.9 |
   | `07_correlation_analysis.py` | Correlation heatmap of raw numeric features | Figure 4.10 |
   | `08_feature_correlation.py` | Engineered feature correlation matrix | Figure 4.11 |
   | `09_feature_importance_preliminary.py` | Quick tree-based importance before formal modeling | Figure 4.12 |

2. **Generate all 12 EDA figures** (Figures 4.1–4.12):
   - Save as PNG + SVG in `results/figures/eda/`
   - 300 DPI, publication-quality
   - Proper titles, axis labels, legends, color schemes

3. **Copy figures** to `web/media/figures/eda/` for web display

**Supported Clubs for EDA**:
- Arsenal, Liverpool, Manchester City, Chelsea, Manchester United

**Script Template**:
```python
# Each script follows this pattern:
# 1. Import libraries
# 2. Load cleaned dataset from data/processed/
# 3. Generate specific analysis
# 4. Save figures to results/figures/eda/
# 5. Print summary statistics

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import os

# Configuration
RESULTS_DIR = '../../results/figures/eda'
DATA_PATH = '../../data/processed/cleaned_dataset.csv'
DPI = 300
```

**Figures Produced**:

| Figure | Title | Type |
|--------|-------|------|
| 4.1 | Dataset Summary | Table/heatmap |
| 4.2 | Match Outcome Distribution | Bar chart |
| 4.3 | Home Goals Distribution | Histogram |
| 4.4 | Away Goals Distribution | Histogram |
| 4.5 | Average Goals Scored by Club | Horizontal bar |
| 4.6 | Average Goals Conceded by Club | Horizontal bar |
| 4.7 | Home vs Away Performance | Grouped bar |
| 4.8 | Team Form Trend | Line chart |
| 4.9 | Head-to-Head Analysis | Heatmap |
| 4.10 | Correlation Heatmap | Correlation matrix |
| 4.11 | Feature Correlation Matrix | Correlation matrix |
| 4.12 | Feature Importance Before Modeling | Feature importance bar |

**Files Created**:
- `exploratory/01_data_overview.py`
- `exploratory/02_outcome_distribution.py`
- `exploratory/03_goals_analysis.py`
- `exploratory/04_team_performance.py`
- `exploratory/05_form_analysis.py`
- `exploratory/06_h2h_analysis.py`
- `exploratory/07_correlation_analysis.py`
- `exploratory/08_feature_correlation.py`
- `exploratory/09_feature_importance_preliminary.py`
- `results/figures/eda/` (12 PNG + 12 SVG files)

---

### Phase 3: Feature Engineering

**Location**: `model/feature_engineering.py` and `web/ml_engine/feature_engineering.py`

**Tasks**:

1. **Implement all 8 feature groups** — every feature computed using ONLY historical matches:

   | Group | Features | Key Metrics |
   |-------|----------|-------------|
   | 1. Team Form (Last 5) | 10 | Wins, Draws, Losses, Points, Goals Scored, Goals Conceded, Goal Difference, Average Goals, Average Shots, Average Shots on Target |
   | 2. Home Performance | 4 | Home Win Rate, Average Home Goals, Average Home Goals Conceded, Average Home Shots, Average Home Shots on Target |
   | 3. Away Performance | 4 | Away Win Rate, Average Away Goals, Average Away Goals Conceded, Average Away Shots, Average Away Shots on Target |
   | 4. Head-to-Head (Last 5) | 5 | Home Wins, Away Wins, Draws, Goals Scored, Goals Conceded, Goal Difference |
   | 5. Attacking Strength | 4 | Average Goals, Average Shots, Average Shots on Target, Average Corners |
   | 6. Defensive Strength | 5 | Average Goals Conceded, Clean Sheets, Shots Faced, Average Fouls, Average Yellow Cards, Average Red Cards |
   | 7. Discipline | 3 | Average Fouls, Average Yellow Cards, Average Red Cards |
   | 8. Momentum | 3 | Winning Streak, Losing Streak, Unbeaten Streak |

   **Total**: ~38 engineered features

2. **Critical anti-leakage architecture**:
   ```python
   # For each match at index i, compute features using only rows 0..i-1
   for i in range(len(df)):
       historical = df.iloc[:i]  # ONLY past matches
       features[i] = compute_all_features(historical, df.iloc[i])
   ```

3. **Handle edge cases**:
   - First few matches per team with insufficient history → fill with team-wide averages
   - Teams not seen before → fill with overall league averages
   - NaN from division by zero → fill with 0

4. **Generate outputs**:
   - `results/engineered_features/engineered_features.csv`
   - `results/feature_reports/feature_descriptions.csv`
   - `results/feature_reports/feature_statistics.csv`
   - `results/feature_reports/feature_dictionary.xlsx`

5. **Generate Feature Engineering Report** (`results/feature_reports/Feature Engineering Report.pdf`):
   - Complete explanation of every engineered feature
   - Mathematical equations where applicable
   - Feature dependency diagrams
   - Sample calculations
   - Data flow diagrams
   - Suitable for direct inclusion in Chapter Three of the dissertation

**Feature Descriptions CSV Format**:

| Feature Name | Description | Formula | Input Variables | Output Type | Purpose | Example Calculation |
|-------------|-------------|---------|----------------|-------------|---------|-------------------|
| form_wins_last5 | Number of wins in last 5 matches | count(result == 'W' for last 5) | FTR, Date, HomeTeam, AwayTeam | int | Captures recent winning form | 3 wins in last 5 → 3 |

**Files Created**:
- `model/feature_engineering.py` (standalone, for EDA/training)
- `web/ml_engine/feature_engineering.py` (Django-integrated version)
- `results/engineered_features/engineered_features.csv`
- `results/feature_reports/feature_descriptions.csv`
- `results/feature_reports/feature_statistics.csv`
- `results/feature_reports/feature_dictionary.xlsx`
- `results/feature_reports/Feature Engineering Report.pdf`

---

### Phase 4: Feature Selection

**Location**: `model/feature_selection.py` and `web/ml_engine/feature_selection.py`

**Tasks**:

1. **Correlation Analysis**:
   - Compute Pearson correlation matrix on all engineered features
   - Identify pairs with |r| > 0.90
   - Drop one feature from each highly correlated pair (keep the one with higher SHAP importance)

2. **SHAP Feature Importance**:
   - Train a preliminary CatBoost model on all engineered features
   - Compute SHAP values using `shap.TreeExplainer`
   - Rank features by mean |SHAP value|
   - Select top features (typically 20–30)

3. **Export**:
   - `results/tables/selected_features.csv` — final feature list with ranks

4. **Visualizations**:
   - Figure 4.11: Feature Correlation Matrix
   - Figure 4.12: Feature Importance Before Modeling

**Files Created**:
- `model/feature_selection.py`
- `web/ml_engine/feature_selection.py`
- `results/tables/selected_features.csv`

---

### Phase 5: Model Training (Stacking Ensemble)

**Location**: `model/train_models.py` and `web/ml_engine/model_training.py`

**Tasks**:

1. **Train/Test Split** (chronological, NO random shuffle):
   ```
   Training: 2019-2020 through 2023-2024 (~1,900 matches)
   Testing:  2024-2025 (380 matches)
   ```

2. **Base Learners** (Level 0):

   | Model | Library | Key Parameters |
   |-------|---------|---------------|
   | CatBoost | `catboost` | `iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=0, loss_function='MultiClass'` |
   | LightGBM | `lightgbm` | `n_estimators=1000, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1, objective='multiclass'` |
   | XGBoost | `xgboost` | `n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0, objective='multi:softprob'` |

3. **Meta Learner** (Level 1):
   ```python
   from sklearn.linear_model import LogisticRegression
   meta_model = LogisticRegression(
       multi_class='multinomial',
       solver='lbfgs',
       max_iter=1000,
       random_state=42
   )
   ```

4. **Stacking Pipeline**:
   ```
   Input Features (X)
       ├── CatBoost  → 3 class probabilities (p_home, p_draw, p_away)
       ├── LightGBM  → 3 class probabilities (p_home, p_draw, p_away)
       └── XGBoost   → 3 class probabilities (p_home, p_draw, p_away)
   Concatenated → 9 features
       └── Logistic Regression → Final prediction + 3 probabilities
   ```

5. **Save all models** to `web/trained_models/`:
   - `catboost_model.cbm`
   - `lightgbm_model.txt`
   - `xgboost_model.json`
   - `logistic_meta_model.pkl`
   - `feature_pipeline.pkl` (selected features list, any scalers)
   - `selected_features.pkl`

**Reproducibility**:
- `random_state=42` / `seed=42` everywhere
- Document Python version and library versions in README

**Files Created**:
- `model/train_models.py`
- `model/utils.py`
- `web/ml_engine/model_training.py`
- `web/trained_models/catboost_model.cbm`
- `web/trained_models/lightgbm_model.txt`
- `web/trained_models/xgboost_model.json`
- `web/trained_models/logistic_meta_model.pkl`
- `web/trained_models/feature_pipeline.pkl`

---

### Phase 6: Model Evaluation

**Location**: `model/evaluate.py` and `web/ml_engine/evaluation.py`

**Tasks**:

1. **Compute all metrics** on test set:

   | Metric | Description | Library |
   |--------|-------------|---------|
   | Accuracy | Overall correct predictions | `sklearn.metrics.accuracy_score` |
   | Precision | Per-class precision (macro + weighted) | `sklearn.metrics.precision_score` |
   | Recall | Per-class recall (macro + weighted) | `sklearn.metrics.recall_score` |
   | F1 Score | Per-class F1 (macro + weighted) | `sklearn.metrics.f1_score` |
   | ROC-AUC | One-vs-rest AUC for 3 classes | `sklearn.metrics.roc_auc_score` |
   | Log Loss | Multiclass log loss | `sklearn.metrics.log_loss` |
   | Balanced Accuracy | Accounts for class imbalance | `sklearn.metrics.balanced_accuracy_score` |

2. **Evaluate each base model AND the stacking ensemble**:
   - CatBoost alone
   - LightGBM alone
   - XGBoost alone
   - Stacking Ensemble (final)

3. **Generate model evaluation figures** (Figures 4.13–4.19):

   | Figure | Title | Type |
   |--------|-------|------|
   | 4.13 | Accuracy Comparison | Grouped bar chart |
   | 4.14 | Precision Comparison | Grouped bar chart |
   | 4.15 | Recall Comparison | Grouped bar chart |
   | 4.16 | F1 Score Comparison | Grouped bar chart |
   | 4.17 | ROC Curves | Line plot (per-class, all models overlaid) |
   | 4.18 | Confusion Matrix | Heatmap |
   | 4.19 | Log Loss Comparison | Bar chart |

4. **Export**:
   - `results/tables/evaluation_metrics.csv`
   - `results/tables/classification_report.csv`
   - All figures to `results/figures/model/`

**Files Created**:
- `model/evaluate.py`
- `web/ml_engine/evaluation.py`
- `results/tables/evaluation_metrics.csv`
- `results/tables/classification_report.csv`
- `results/figures/model/` (7 PNG + 7 SVG files)

---

### Phase 7: Model Explainability (SHAP)

**Location**: `model/explainability.py` and `web/ml_engine/explainability.py`

**Tasks**:

1. **Compute SHAP values** for the stacking ensemble:
   - Use `shap.TreeExplainer` for each base model (CatBoost, LightGBM, XGBoost)
   - Aggregate SHAP contributions or use the meta-model's linear coefficients

2. **Generate explainability figures**:

   | Figure | Title | Type |
   |--------|-------|------|
   | 4.20 | SHAP Summary Plot | Beeswarm plot |
   | 4.21 | SHAP Waterfall Plot | Single prediction explanation |
   | 4.22 | Final Feature Importance | Horizontal bar chart from SHAP |

3. **Store SHAP feature ranking** for use in Django prediction page:
   - When a user gets a prediction, show the top 5 SHAP features that influenced it
   - Save feature importance mapping to `web/trained_models/shap_feature_importance.pkl`

4. **Save all figures** to `results/figures/explainability/`

**Files Created**:
- `model/explainability.py`
- `web/ml_engine/explainability.py`
- `results/figures/explainability/` (3 PNG + 3 SVG files)
- `web/trained_models/shap_feature_importance.pkl`

---

### Phase 8: Django Web Application

**Location**: `web/`

**This is the largest phase.** All frontend uses Tailwind CSS with a premium, custom design.

#### 8a. URL Structure

| URL | View | Template |
|-----|------|----------|
| `/` | `core.views.home` | `core/home.html` |
| `/prediction/` | `prediction.views.predict` | `prediction/prediction.html` |
| `/about/` | `core.views.about` | `core/about.html` |
| `/model-info/` | `core.views.model_info` | `core/model_info.html` |
| `/dataset-info/` | `core.views.dataset_info` | `core/dataset_info.html` |
| `/methodology/` | `core.views.methodology` | `core/methodology.html` |
| `/history/` | `prediction.views.history` | `prediction/history.html` |
| `/features/` | `ml_engine.views.features` | `ml_engine/features.html` |
| `/reports/` | `reports.views.index` | `reports/reports.html` |
| `/admin/` | Django admin | Built-in |

#### 8b. Prediction Page (Core Feature)

**User interaction flow**:
1. User selects Home Team from dropdown (5 supported clubs)
2. User selects Away Team from dropdown
3. User clicks "Predict" button
4. AJAX request (no page reload) → loading spinner
5. Backend:
   - Loads historical matches for both teams from cleaned dataset
   - Runs feature engineering pipeline
   - Feeds features through stacking ensemble
   - Returns JSON response with prediction + probabilities + SHAP features
6. Frontend displays prediction result card with:
   - Predicted result (Home Win / Draw / Away Win) with color-coded badge
   - Confidence percentage with animated progress bar
   - Three probability bars (Home Win %, Draw %, Away Win %)
   - Top 5 SHAP features influencing this specific prediction
   - Animated transitions

#### 8c. Django Models

```python
# prediction/models.py
class PredictionHistory(models.Model):
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    predicted_result = models.CharField(max_length=10)
    predicted_label = models.CharField(max_length=20)  # Human-readable
    confidence = models.FloatField()
    prob_home = models.FloatField()
    prob_draw = models.FloatField()
    prob_away = models.FloatField()
    shap_features = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.home_team} vs {self.away_team} → {self.predicted_label}"


# dataset/models.py
class DatasetInfo(models.Model):
    filename = models.CharField(max_length=200)
    season = models.CharField(max_length=20)
    rows = models.IntegerField()
    columns = models.IntegerField()
    date_range_start = models.DateField()
    date_range_end = models.DateField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.season} ({self.filename})"


# ml_engine/models.py
class ModelMetrics(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision_macro = models.FloatField()
    recall_macro = models.FloatField()
    f1_macro = models.FloatField()
    roc_auc = models.FloatField()
    log_loss_value = models.FloatField()
    balanced_accuracy = models.FloatField()
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} — Acc: {self.accuracy:.4f}"


class FeatureInfo(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    feature_group = models.CharField(max_length=100)
    importance_rank = models.IntegerField(null=True, blank=True)
    shap_importance = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['importance_rank']

    def __str__(self):
        return self.name
```

#### 8d. Premium Tailwind Design

**Design System**:
- **Color palette**: Dark navy primary (#0f172a), emerald accent (#10b981), gold highlights (#f59e0b)
- **Typography**: Inter font family (Google Fonts)
- **Components**: Glassmorphism cards, gradient buttons, animated progress bars
- **Dark theme** as default with light mode toggle
- **Responsive**: Mobile-first, works on all screen sizes
- **Icons**: Heroicons via SVG inline

**Base Template Structure** (`templates/base.html`):
```html
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Football Prediction System{% endblock %}</title>
    <!-- Tailwind CSS (compiled by django-tailwind-cli) -->
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/dist/tailwind.css' %}">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{% static 'css/custom.css' %}">
    <!-- Inter font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
</head>
<body class="bg-slate-950 text-white font-['Inter'] min-h-screen">
    <!-- Navigation -->
    <nav>...</nav>

    <!-- Main Content -->
    <main class="container mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>

    <!-- Footer -->
    <footer>...</footer>

    <!-- JavaScript -->
    <script src="{% static 'js/prediction.js' %}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

**Prediction Page Design**:
- Two team selector dropdowns with team crests/icons
- "Predict" button with gradient and hover animation
- Result card appears with slide-in animation
- Confidence shown as animated circular progress
- Probability bars with color coding (green for Home, yellow for Draw, red for Away)
- SHAP features shown as horizontal bar chart below result

#### 8e. Prediction Service (`web/prediction/ml_service.py`)

```python
class PredictionService:
    """Singleton service that loads models once and serves predictions."""

    def __init__(self):
        self.catboost_model = None
        self.lightgbm_model = None
        self.xgboost_model = None
        self.meta_model = None
        self.selected_features = None
        self.dataset = None

    def load_models(self):
        """Load all trained models from disk."""
        # Lazy loading — called once on first prediction

    def predict(self, home_team, away_team):
        """
        Generate prediction for a match.

        Returns:
            dict: {
                'predicted_result': 'H'/'D'/'A',
                'predicted_label': 'Home Win'/'Draw'/'Away Win',
                'confidence': 0.85,
                'probabilities': {'home': 0.85, 'draw': 0.10, 'away': 0.05},
                'shap_features': [
                    {'name': 'form_points_last5', 'value': 12, 'importance': 0.15},
                    ...
                ]
            }
        """
```

#### 8f. Admin Panel

Customize Django admin for all models:
- `PredictionHistory`: search by team, filter by date, display all probability fields
- `DatasetInfo`: list view with season, row count
- `ModelMetrics`: side-by-side model comparison
- `FeatureInfo`: sortable by importance rank

**Files Created**:
- `web/core/views.py`, `web/core/urls.py`, `web/core/admin.py`
- `web/prediction/views.py`, `web/prediction/urls.py`, `web/prediction/forms.py`, `web/prediction/ml_service.py`, `web/prediction/admin.py`
- `web/ml_engine/views.py`, `web/ml_engine/urls.py`, `web/ml_engine/admin.py`
- `web/dataset/views.py`, `web/dataset/urls.py`, `web/dataset/admin.py`
- `web/reports/views.py`, `web/reports/urls.py`
- `web/visualization/views.py`, `web/visualization/urls.py`
- `web/templates/base.html`
- `web/templates/core/home.html`
- `web/templates/core/about.html`
- `web/templates/core/model_info.html`
- `web/templates/core/dataset_info.html`
- `web/templates/core/methodology.html`
- `web/templates/prediction/prediction.html`
- `web/templates/prediction/history.html`
- `web/templates/ml_engine/features.html`
- `web/templates/reports/reports.html`
- `web/static/css/custom.css`
- `web/static/js/prediction.js`

---

### Phase 9: Report Exports

**Location**: `web/reports/`

**Tasks**:

1. **Create `web/reports/exporters.py`** with functions to generate:
   - Evaluation metrics → CSV, Excel (openpyxl), PDF (reportlab)
   - Prediction history → CSV, Excel
   - Feature engineering reports → CSV, Excel
   - Classification reports → CSV, Excel

2. **Create download views**:
   - `/reports/evaluation/` — download evaluation metrics
   - `/reports/predictions/` — download prediction history
   - `/reports/features/` — download feature reports
   - `/reports/download/<format>/` — generic download endpoint

3. **Report formats**:

   | Report | CSV | Excel | PDF |
   |--------|-----|-------|-----|
   | Evaluation Metrics | ✓ | ✓ | ✓ |
   | Classification Report | ✓ | ✓ | ✓ |
   | Prediction History | ✓ | ✓ | — |
   | Feature Descriptions | ✓ | ✓ | ✓ |
   | Feature Statistics | ✓ | ✓ | ✓ |
   | Selected Features | ✓ | ✓ | — |

**Files Created**:
- `web/reports/exporters.py`
- Updated `web/reports/views.py`
- Updated `web/reports/urls.py`

---

### Phase 10: Documentation + Deployment

**Tasks**:

1. **Create documentation files**:

   | File | Location | Content |
   |------|----------|---------|
   | `README.md` | Root | Project overview, quick start, screenshots |
   | `INSTALLATION_GUIDE.md` | Root | Step-by-step local setup |
   | `DEPLOYMENT_GUIDE.md` | Root | Render deployment walkthrough |
   | `DEVELOPER_GUIDE.md` | Root | Architecture, code conventions, extending |
   | `exploratory/README.md` | `exploratory/` | How to run EDA scripts |
   | `model/README.md` | `model/` | How to retrain models |
   | `web/README.md` | `web/` | Django project overview |

2. **Render deployment files**:

   ```yaml
   # render.yaml
   databases:
     - name: football-db
       plan: free
       databaseName: football_prediction
       user: football_user

   services:
     - type: web
       plan: free
       name: football-prediction
       runtime: python
       buildCommand: ./build.sh
       startCommand: 'gunicorn config.wsgi:application'
       envVars:
         - key: DATABASE_URL
           fromDatabase:
             name: football-db
             property: connectionString
         - key: SECRET_KEY
           generateValue: true
         - key: DEBUG
           value: "false"
         - key: WEB_CONCURRENCY
           value: "4"
   ```

   ```bash
   #!/usr/bin/env bash
   # build.sh
   set -o errexit

   pip install -r requirements.txt
   python manage.py tailwind build
   python manage.py collectstatic --no-input
   python manage.py migrate
   ```

3. **`requirements.txt`** (web/):
   ```
   Django>=5.0,<6.0
   django-tailwind-cli>=4.0
   whitenoise[brotli]>=6.0
   gunicorn>=22.0
   dj-database-url>=2.0
   psycopg2-binary>=2.9
   pandas>=2.0
   numpy>=1.24
   scikit-learn>=1.3
   catboost>=1.2
   lightgbm>=4.0
   xgboost>=2.0
   shap>=0.43
   matplotlib>=3.7
   seaborn>=0.12
   openpyxl>=3.1
   reportlab>=4.0
   joblib>=1.3
   ```

**Files Created**:
- Root `README.md`
- Root `INSTALLATION_GUIDE.md`
- Root `DEPLOYMENT_GUIDE.md`
- Root `DEVELOPER_GUIDE.md`
- `exploratory/README.md`
- `model/README.md`
- `web/README.md`

---

## Complete Execution Order

```
Phase 0: Scaffolding
    │
    ▼
Phase 1: Data Pipeline
    │
    ├──► Phase 2: EDA Scripts (exploratory/)
    │
    ▼
Phase 3: Feature Engineering (model/)
    │
    ▼
Phase 4: Feature Selection
    │
    ▼
Phase 5: Model Training
    │
    ├──► Phase 6: Model Evaluation
    │
    └──► Phase 7: SHAP Explainability
              │
              ▼
Phase 8: Django Web Application
    │
    ├──► Phase 9: Report Exports
    │
    └──► Phase 10: Documentation + Render Deploy
```

---

## Key Technical Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Date format inconsistency across CSVs | `pd.to_datetime(dayfirst=True, errors='coerce')` with fallback parsing |
| Missing values in match stats | Fill numeric NaNs with 0 (legitimate — means 0 shots, 0 fouls, etc.) |
| Feature engineering data leakage | Strict chronological indexing — `df.iloc[:i]` only for row `i` |
| Teams with few historical matches | Use team-wide season averages as fallback |
| Class imbalance (fewer draws) | Report balanced accuracy; consider `class_weight='balanced'` |
| Model file sizes | Use `joblib` with compression; lazy-load in Django |
| Column differences across seasons | Standardize column set after merging; drop unmatched columns |
| Render free tier limits | PostgreSQL expires after 30 days on free tier — document upgrade path |
| Tailwind build on Render | `django-tailwind-cli` handles this in `build.sh` — no Node.js needed |

---

## Estimated File Count

| Component | Files |
|-----------|-------|
| Django config + apps (Python) | ~25 |
| EDA scripts | ~9 |
| ML scripts | ~6 |
| HTML templates | ~10 |
| CSS/JS static files | ~3 |
| Results/figures (PNG + SVG) | ~44 |
| Results/tables/reports | ~10 |
| Data files (CSV) | ~8 |
| Documentation | ~8 |
| Config (requirements, Procfile, etc.) | ~5 |
| **Total** | **~128 files** |

---

## Supported Clubs

Only generate predictions for matches involving:
- Arsenal
- Liverpool
- Manchester City
- Chelsea
- Manchester United

Historical matches against every opponent are retained for feature generation.

---

## Target Variable

**FTR** (Full Time Result):
- Home Win = 0
- Draw = 1
- Away Win = 2

---

## Dataset Columns (Retained)

```
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR,
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
```

**All betting odds columns are EXCLUDED** — never used in any part of the system.

---

## Anti-Leakage Rules

1. Never use future information when computing features
2. For each match at index `i`, compute features using only rows `0..i-1`
3. All betting odds columns are dropped at the data loading stage
4. Feature selection is performed on training data only
5. Models are evaluated on unseen 2024–2025 season data
6. No random shuffling — chronological order is always preserved
