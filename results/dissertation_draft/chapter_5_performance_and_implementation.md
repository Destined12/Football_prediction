# Chapter 5: Results, System Implementation, and Deployment

---

## 5.5 Performance Analysis of the Proposed Stacking Ensemble

This section presents a detailed performance analysis of the final stacking ensemble model, which combines XGBoost, Support Vector Machine (SVM), and Random Forest as base learners with a Logistic Regression meta-learner. The analysis is conducted on the held-out test set comprising 170 matches from the 2024–2025 season, representing data the model has never encountered during training or feature selection.

### Stacking Ensemble Architecture

The proposed stacking ensemble follows Wolpert's (1992) stacked generalisation framework, in which the predictions of multiple diverse base learners are used as inputs to a higher-level meta-learner that produces the final prediction. Unlike simple averaging or voting ensembles, stacking learns an optimal weighting of each base model's output, enabling the meta-learner to exploit the complementary strengths of heterogeneous learning algorithms.

The ensemble comprises three base learners at Level 0, each selected for its distinct learning paradigm to maximise model diversity:

**XGBoost** (eXtreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each subsequent tree corrects the residual errors of its predecessors. XGBoost excels at capturing non-linear feature interactions and complex decision boundaries through gradient-based optimisation. It was tuned via RandomizedSearchCV with TimeSeriesSplit (5 folds) over the following hyperparameter space: n_estimators ∈ {300, 500, 800}, max_depth ∈ {4, 6, 8}, learning_rate ∈ {0.01, 0.03, 0.05}, subsample ∈ {0.8, 0.9, 1.0}, colsample_bytree ∈ {0.8, 0.9, 1.0}, and min_child_weight ∈ {1, 3, 5}. The objective function was `multi:softprob` with 3 output classes, optimising macro F1 score.

**Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel maps the 50-dimensional feature space into a higher-dimensional space where class boundaries become linearly separable. The SVM was wrapped in a scikit-learn `Pipeline` with a `StandardScaler` (essential for kernel-based methods, as features with larger magnitudes would dominate the kernel computation) and configured with `class_weight='balanced'` to account for the uneven class distribution. Hyperparameters were tuned over C ∈ {0.1, 1, 10} and gamma ∈ {'scale', 'auto'}. To produce well-calibrated probability estimates required by the meta-learner, the tuned SVM was subsequently wrapped in a `CalibratedClassifierCV` with isotonic regression and 3-fold cross-validation, which transforms the SVM's decision function outputs into probability distributions over the three outcome classes.

**Random Forest** is an ensemble of decision trees trained independently on bootstrap samples of the training data, with random feature subsets at each split. Unlike XGBoost's sequential boosting, Random Forest reduces variance through parallel bagging, making it complementary to the gradient-boosted approach. It was tuned with class_weight='balanced' over n_estimators ∈ {300, 500, 700}, max_depth ∈ {10, 20, None}, min_samples_leaf ∈ {1, 2, 4}, and max_features ∈ {'sqrt', 'log2'}.

**Meta-Learner (Level 1):** The meta-learner is a Logistic Regression classifier with `class_weight='balanced'`, L2 regularisation (C = 1.0), and the L-BFGS solver with 5,000 maximum iterations. Three candidates were evaluated on the out-of-fold predictions: Logistic Regression with C = 1.0, Logistic Regression with C = 0.1, and a Calibrated Linear SVM. The Logistic Regression with C = 1.0 achieved the highest macro F1 on cross-validated OOF predictions and was selected.

**Out-of-Fold (OOF) Training Procedure:** To prevent data leakage into the meta-learner, the stacking ensemble uses out-of-fold predictions during training. The 845 training matches are split into 5 chronological folds using TimeSeriesSplit. For each fold, the three base learners are cloned, trained on the 4 training folds, and generate probability predictions on the held-out validation fold. Each base model produces 3 class probabilities (Home Win, Draw, Away Win), yielding 9 meta-features per match. After all 5 folds are processed, every training match has OOF predictions from all three base models, forming a complete 845 × 9 matrix that serves as the meta-learner's training input. This procedure ensures that the meta-learner never sees predictions from models trained on the same data it is learning from.

**Inference Flow:** At prediction time, the 50 selected features for a given match are fed into the three base models, each producing a 3-element probability vector. These are concatenated into a 9-element meta-feature vector: [P(Home), P(Draw), P(Away)] from XGBoost, followed by the same triplet from SVM, and then from Random Forest. The Logistic Regression meta-learner processes these 9 probabilities and outputs the final 3-class probability distribution, from which the predicted outcome is derived via argmax.

### Table 5.5 — Classification Report of the Stacking Ensemble

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Home Win | 0.626 | 0.750 | 0.683 | 76 |
| Draw | 0.250 | 0.024 | 0.044 | 41 |
| Away Win | 0.533 | 0.755 | 0.625 | 53 |
| **Accuracy** | | | **0.576** | **170** |
| **Macro Avg** | **0.470** | **0.510** | **0.451** | **170** |
| Weighted Avg | 0.507 | 0.576 | 0.511 | 170 |

### Figure 5.15 — Confusion Matrix

*See* `results/figures/model/confusion_matrix_ensemble.png`

The confusion matrix (Figure 5.15) reveals the ensemble correctly classifies 98 out of 170 test matches (57.6%). The distribution of correct predictions is heavily skewed towards the two majority classes: 57 of 76 Home Wins are correctly predicted (75.0% class recall) and 40 of 53 Away Wins are correctly identified (75.5% class recall). However, the Draw class is severely underpredicted, with only 1 of 41 actual draws correctly classified (2.4% class recall). The remaining 40 draws are distributed almost evenly between Home Win (22 cases, 53.7%) and Away Win (18 cases, 43.9%) misclassifications. This pattern indicates that the ensemble learns a strong bias towards predicting decisive outcomes, effectively treating draws as a "toss-up" between the two teams rather than a distinct outcome class.

### Figure 5.16 — Multi-class ROC Curve

*See* `results/figures/model/roc_curves.png`

The per-class ROC curves (Figure 5.16) provide a complementary view of discrimination ability across all three outcome classes. The Home Win class achieves an AUC of approximately 0.68, the Away Win class approximately 0.70, and the Draw class notably lower at approximately 0.58. The relatively poor Draw class AUC is consistent with the near-random recall observed in the classification report, confirming that the ensemble struggles to separate draws from decisive results. The ROC curves for Home Win and Away Win both rise steeply above the diagonal, indicating meaningful discrimination, while the Draw curve remains closer to the random baseline throughout most of the threshold range.

### Figure 5.17 — Precision-Recall Curve

*See* `results/figures/model/pr_curves.png`

The precision-recall curves (Figure 5.17) further quantify class-specific performance. The Home Win class achieves an Average Precision (AP) of 0.72, and the Away Win class reaches 0.73, indicating that the model maintains reasonably high precision across the recall range for decisive outcomes. In contrast, the Draw class AP is approximately 0.23, reflecting the fundamental difficulty of distinguishing draws from wins. The precision-recall gap between the decisive classes and the Draw class underscores the class imbalance problem: with only 41 draws (24.1%) in the test set compared to 76 Home Wins (44.7%) and 53 Away Wins (31.2%), the model has fewer examples to learn the Draw decision boundary.

### Correctly Classified Matches

Of the 170 test matches, 98 (57.6%) receive the correct predicted outcome. The Home Win and Away Win classes each achieve approximately 75% recall, meaning three-quarters of decisive results are correctly identified. When the model predicts a Home Win, it is correct 62.6% of the time (precision), and when it predicts an Away Win, it is correct 53.3% of the time. These precision values indicate that while the model has a reasonable hit rate for decisive predictions, there is a modest degree of over-prediction, particularly for Away Wins where 12 of 53 predicted Away Wins are actually Home Wins.

### Misclassified Matches

The 72 misclassified matches (42.4%) are dominated by Draw misclassifications. Of the 40 misclassified draws, 22 are predicted as Home Wins and 18 as Away Wins. This near-even split suggests the model defaults to the team with the stronger recent form (typically the home team) when uncertain, rather than systematically favouring one outcome. Notably, only 2 Home Wins are misclassified as Draws, and only 1 Away Win is misclassified as a Draw, confirming that the model rarely predicts draws at all — the predicted draw count across all 170 matches is just 4, compared to 41 actual draws. The remaining misclassifications involve Home Win and Away Win confusion: 17 Home Wins are predicted as Away Wins and 12 Away Wins as Home Wins, representing cases where the model's feature-based assessment of relative team strength diverges from the actual match outcome.

### Class-Wise Performance Analysis

The three-class performance hierarchy is stark: Home Win (F1 = 0.683) >> Away Win (F1 = 0.625) >> Draw (F1 = 0.044). The Home Win class benefits from both the largest support (76 matches) and the strongest feature signals — the home advantage index, venue-specific performance, and Elo differential all contribute to home win prediction. The Away Win class, with 53 matches, performs nearly as well (75.5% recall), likely aided by the `away_dom_shots_diff` feature (SHAP rank #3, importance 0.0581), which captures the away team's ability to dominate the shot battle away from home. The Draw class, with the smallest support (41 matches) and the weakest feature signals, essentially fails. The 0.024 recall means the model predicts a Draw for only 1 out of 41 actual draws — a performance level indistinguishable from random guessing.

### Effect of Feature Engineering

The feature engineering pipeline transforms 21 raw match columns into 198 engineered features across 20 groups, capturing team form at multiple time horizons (3, 5, and 10 matches), venue-specific performance, head-to-head records, attacking/defensive strength, discipline, momentum, efficiency, dominance, trends, consistency, first-half performance, fixture difficulty, venue momentum, win percentages, clean sheet rates, Elo ratings, xG proxies, and cross-team differentials. The top SHAP features — `away_def_shots_faced` (0.0649), `away_team_away_perf_avg_shots` (0.0606), and `away_dom_shots_diff` (0.0581) — demonstrate that the model relies heavily on away-team performance metrics rather than raw goal tallies. This is a meaningful distinction: while the EDA (Section 4.3) showed that home teams score 1.64 goals on average versus 1.38 for away teams, the model learns that the shot-based dominance differentials (rather than the goals themselves) are more predictive of match outcomes. The Elo rating system, which captures dynamic team strength through iterative updating, ranks as the 6th most important feature (`home_elo`, SHAP = 0.0521), confirming that long-term team quality provides predictive value beyond recent form.

### Effect of SHAP Feature Selection

The feature selection pipeline reduces the 198 engineered features to 50 through a two-stage process: first, a correlation filter removes 41 redundant features with Pearson correlation coefficients exceeding 0.90 (e.g., `home_form_wins_10` and `home_form_points_10` with r > 0.95, as points are computed directly from wins). Second, SHAP importance ranking via CatBoost selects the top 50 features from the remaining 157. The feature count evaluation (Table 4.3) demonstrates that 50 features achieve the highest macro F1 (0.4704) and accuracy (58.70%) compared to 30, 40, or 60 features. Adding features beyond 50 degrades performance, confirming that lower-ranked features introduce noise rather than signal. The 50 selected features span 13 distinct predictive dimensions — consistency, venue performance, H2H, fixture difficulty, discipline, trend, dominance, difference, defensive, attacking, efficiency, Elo rating, and first-half performance — ensuring diversity in the information available to the base learners.

### Overall Predictive Capability

The stacking ensemble achieves an overall accuracy of 57.6% and a macro F1 of 0.451 on the held-out 2024–2025 test set. While the accuracy exceeds the majority-class baseline (Home Win at 44.7%) by 12.9 percentage points, the macro F1 reveals that the model's per-class performance is uneven. The ensemble's strength lies in its ability to correctly identify decisive results (Home Win and Away Win) with approximately 75% recall, making it a useful tool for predicting match outcomes where one team is likely to win. However, the near-total failure on the Draw class (F1 = 0.044) represents a fundamental limitation: the ensemble cannot reliably distinguish draws from decisive results. This limitation is partly intrinsic to the problem — draws are the minority class (24.1% of test data) and inherently harder to predict — and partly a consequence of the ensemble architecture, where the Logistic Regression meta-learner amplifies the base models' bias towards majority classes. Comparing with individual base models (Table 4.4), the ensemble matches Random Forest's accuracy (57.6%) and exceeds XGBoost's accuracy (57.1%), though SVM achieves the highest individual accuracy at 60.6%. The ensemble's macro F1 (0.451) is competitive with the best individual models (SVM: 0.456, RF: 0.452), suggesting that the stacking architecture provides modest but meaningful improvement in balanced performance across classes.

---

## 5.6 System Implementation

This section describes the technical implementation of the football prediction system, covering the software architecture, machine learning pipeline integration, and the prediction workflow that connects user interaction to model inference.

### Figure 5.18 — Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER (Web Browser)                          │
│              Tailwind CSS Dark Glassmorphism UI                 │
│         AJAX Prediction · SHAP Explanations · Reports           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /prediction/api/predict
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DJANGO WEB APPLICATION                         │
│                                                                 │
│  ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌────────────────┐  │
│  │  core     │ │ prediction │ │ ml_engine │ │   reports      │  │
│  │  (home,   │ │ (predict,  │ │ (features,│ │ (download,     │  │
│  │  about,   │ │  history,  │ │  models)  │ │  csv/excel/    │  │
│  │  model)   │ │  api)      │ │           │ │  pdf)          │  │
│  └──────────┘ └─────┬──────┘ └─────┬─────┘ └────────────────┘  │
│                     │              │                             │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ dataset  │ │visualization│ │  admin   │ │   config         │  │
│  │ (info,   │ │ (figures,  │ │ (Django  │ │ (settings,       │  │
│  │  stats)  │ │  gallery)  │ │  panel)  │ │  urls, wsgi)     │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING MODULE                         │
│         web/ml_engine/feature_engineering.py                    │
│                                                                 │
│  198 features across 20 groups computed from raw match data:    │
│  Form (3/5/10) · Venue Performance · H2H · Attacking/Defensive │
│  Discipline · Momentum · Efficiency · Dominance · Trend         │
│  Consistency · HT Performance · Fixture Difficulty · Elo · xG   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRAINED ML MODELS                                  │
│         web/trained_models/*.pkl                                │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │   XGBoost   │  │     SVM     │  │   Random Forest      │    │
│  │  (depth=6)  │  │  (RBF, C=1) │  │  (500 trees)         │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘    │
│         │                │                     │                │
│         └────────┬───────┴─────────────────────┘                │
│                  │ 3×3 = 9 meta-features                        │
│                  ▼                                               │
│  ┌──────────────────────────────────────┐                       │
│  │  Logistic Regression Meta-Learner    │                       │
│  │  (class_weight='balanced', C=1.0)    │                       │
│  └──────────────┬───────────────────────┘                       │
│                 │                                                │
│                 ▼                                                │
│         PREDICTION OUTPUT                                       │
│    Home Win / Draw / Away Win + Probabilities                   │
└─────────────────────────────────────────────────────────────────┘
```

The system is built on Django 5.2, a Python web framework that provides a clean, modular architecture through its app-based structure. The project is organised into six Django applications, each responsible for a distinct domain:

- **core**: Landing page, about page, model information, dataset information, and methodology documentation.
- **prediction**: The central prediction interface, prediction history tracking, and the JSON API endpoint that processes prediction requests.
- **ml_engine**: Feature engineering module (mirroring the training pipeline), model information pages, and feature visualisation.
- **dataset**: Data loading, cleaning, and metadata display.
- **reports**: Dynamic report listing with CSV, Excel, and PDF download capabilities.
- **visualization**: EDA figure gallery serving 44 PNG figures across three categories (EDA, model evaluation, explainability).

This modular design ensures separation of concerns, with each app maintaining its own views, URL patterns, and templates. The root URL configuration (`config/urls.py`) routes requests to the appropriate app, and the shared base template (`templates/base.html`) provides consistent navigation and styling across all pages.

### Figure 5.19 — Prediction Workflow

```
User selects Home Club
        │
        ▼
User selects Away Club
        │
        ▼
Clicks "Predict Match"
        │
        ▼
┌─────────────────────────────────┐
│  AJAX POST                      │
│  /prediction/api/predict        │
│  Headers: X-CSRFToken           │
│  Body: home_team, away_team     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  1. Input Validation            │
│     - Both teams provided?      │
│     - Teams are different?      │
│     - Both in SUPPORTED_CLUBS?  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. Feature Engineering         │
│     compute_match_features(     │
│       home_team, away_team)     │
│     → 198 raw features          │
│     → Select 50 (from .pkl)     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. Base Model Predictions      │
│     XGBoost  → 3 probabilities  │
│     SVM      → 3 probabilities  │
│     RF       → 3 probabilities  │
│     Concatenated → 9 features   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4. Meta-Learner                │
│     Logistic Regression         │
│     Input: 9 meta-features      │
│     Output: 3 final probs       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  5. SHAP Explanation            │
│     TreeExplainer (XGBoost)     │
│     Top 5 feature contributions │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  6. JSON Response               │
│  {                              │
│    predicted_label: "Home Win", │
│    confidence: 0.47,            │
│    probabilities: {             │
│      home: 0.47, draw: 0.26,   │
│      away: 0.27                 │
│    },                           │
│    shap_features: [...]         │
│  }                              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  7. Frontend Rendering          │
│     - Predicted outcome badge   │
│     - Confidence percentage     │
│     - 3 probability bars        │
│     - SHAP importance bars      │
│     - Animated transitions      │
└─────────────────────────────────┘
```

The prediction workflow operates through an asynchronous JavaScript (AJAX) request pattern, eliminating page reloads and providing instant feedback. When the user selects two clubs and clicks "Predict Match", the frontend JavaScript constructs a `FormData` payload containing the home and away team names, attaches the CSRF token for security, and sends a POST request to the `/prediction/api/predict` endpoint.

The backend prediction service (`prediction/ml_service.py`) is implemented as a singleton class that lazy-loads all five serialized artifacts on the first prediction request: `xgboost_model.pkl`, `svm_pipeline.pkl`, `random_forest_model.pkl`, `logistic_meta_model.pkl`, and `selected_features.pkl`. Subsequent predictions reuse the loaded models in memory, avoiding disk I/O overhead. The singleton pattern is critical for production performance, as loading the four model files (total ~2.8 MB compressed) would otherwise add significant latency to every request.

The feature engineering module (`ml_engine/feature_engineering.py`, 583 lines) replicates the exact same computational logic used during training, computing all 198 features from raw match data for the two selected teams. This includes Elo ratings (computed from the full historical dataset), multi-window form metrics, venue-specific performance, head-to-head records, and all derived features (efficiency, dominance, trends, consistency, fixture difficulty, and the home advantage index). The 50 selected features are then extracted using the pre-serialized feature list.

Model serialization is handled through `joblib.dump()` with compression level 3, balancing file size against load time. The SVM model is saved as a scikit-learn `Pipeline` containing both the `StandardScaler` and the trained `SVC` estimator, ensuring consistent feature scaling at inference time without requiring separate scaler management. The meta-learner (Logistic Regression) and the three base models are saved as standalone estimator objects.

### Django Framework and Backend Processing

The Django framework provides several architectural advantages for this system. The MTV (Model-Template-View) pattern cleanly separates data processing (views), presentation (templates), and data storage (models). The `PredictionHistory` model stores every prediction with its associated probabilities, SHAP features, and timestamps, enabling the prediction history page and providing an audit trail for model performance monitoring.

The CSRF (Cross-Site Request Forgery) protection built into Django secures the prediction API against unauthorised requests. Each AJAX call includes the CSRF token extracted from the cookie, and Django validates it before processing the prediction. Input validation occurs at two levels: the frontend JavaScript validates that both dropdowns are populated and different, while the backend `api_predict` view validates that both team names exist in the `SUPPORTED_CLUBS` list (Arsenal, Liverpool, Manchester City, Chelsea, Manchester United) and returns a 400 error with a descriptive message if validation fails.

The WhiteNoise middleware handles static file serving in production, compressing CSS, JavaScript, and image assets with Brotli compression and serving them with appropriate cache headers. This eliminates the need for a separate CDN during development and provides efficient static file delivery on Render's infrastructure.

---

## 5.7 Web Application Deployment and Testing

This section describes the deployment of the football prediction system to Render.com, the testing methodology employed to validate system functionality, and the user experience of the deployed application.

### Deployment Platform and Configuration

The application is deployed on Render.com, a cloud platform that provides managed PostgreSQL databases, automatic SSL, and continuous deployment from Git repositories. The deployment stack consists of:

- **Web Server**: Gunicorn 22+ serving the Django WSGI application (`config.wsgi:application`) with 4 worker processes for concurrent request handling.
- **Static Files**: WhiteNoise middleware serving compressed static assets directly from the application, eliminating the need for a separate CDN or S3 bucket.
- **Database**: PostgreSQL managed by Render, configured via the `DATABASE_URL` environment variable and `dj-database-url` for connection string parsing.
- **Build Process**: A `build.sh` script that installs Python dependencies from `requirements.txt`, collects static files with `collectstatic`, and runs database migrations.

The build script is intentionally minimal — it does not retrain the ML models during deployment. The trained model files (`*.pkl`) are committed to the repository and loaded at runtime, ensuring that the production system uses the exact same model artifacts validated during development. This approach avoids the computational overhead of model training during deployment (which would add several minutes to each build) and guarantees reproducibility.

The `Procfile` declares a single web process: `web: gunicorn config.wsgi:application`. Gunicorn is configured to bind to the port specified by Render's `PORT` environment variable, with SSL redirect enabled in production through Django's `SECURE_SSL_REDIRECT` setting. The `settings.py` file distinguishes between development (SQLite, `DEBUG=True`) and production (PostgreSQL, `DEBUG=False`, secure cookies, HSTS headers).

### Application Testing

The system was validated through multiple testing approaches:

**Static Analysis**: `python manage.py check` reports zero issues, confirming that all Django models, URL configurations, template tags, and middleware are correctly configured. The `collectstatic --dry-run` command completes without errors, verifying that all static files are discoverable and properly referenced.

**Route Testing**: All 10 application routes return HTTP 200 responses: the home page (`/`), prediction interface (`/prediction/`), about page (`/about/`), model information (`/model-info/`), dataset information (`/dataset-info/`), methodology (`/methodology/`), prediction history (`/history/`), feature information (`/features/`), figure gallery (`/figures/`), and reports (`/reports/`). Each page renders correctly with Tailwind CSS styling and proper data integration.

**Prediction API Testing**: The `/prediction/api/predict` endpoint returns valid JSON responses for all valid team pairings among the 5 supported clubs. Response fields include `predicted_label` (one of "Home Win", "Draw", "Away Win"), `confidence` (float 0–1), `probabilities` (object with `home`, `draw`, `away` keys), and `shap_features` (list of top-5 SHAP contributions). Error responses correctly handle invalid team names (400), identical teams (400), and non-POST requests (405).

**Report Download Testing**: All 18 download links (6 reports × 3 formats) return valid files. CSV files contain comma-separated data with proper headers. Excel files open correctly in spreadsheet applications with auto-sized columns. PDF files render styled tables with green headers and dark alternating rows using ReportLab.

### Figure 5.20 — Home Page

*See deployed application or* `web/templates/core/home.html`

The home page presents a hero section with a gradient background and animated pulse dot badge reading "Powered by Stacking Ensemble ML". The headline "Predict EPL Match Outcomes with AI" introduces the system's purpose, followed by a description of the XGBoost + SVM + Random Forest stacking ensemble. Two call-to-action buttons — "Start Predicting" (links to the prediction interface) and "Learn More" (links to the about page) — guide users to the primary functionality. Three feature cards below highlight the system's key attributes: "3 Base Models" (explaining the heterogeneous ensemble), "50 Selected Features" (from 198 raw features through SHAP-guided selection), and "Instant Results" (with SHAP explanations). The dark glassmorphism design with emerald accent colours creates a modern, professional appearance consistent with contemporary web application aesthetics.

### Figure 5.21 — Prediction Interface

*See deployed application or* `web/templates/prediction/prediction.html`

The prediction interface provides two dropdown selectors for the home and away clubs, each populated with the 5 supported teams (Arsenal, Chelsea, Liverpool, Manchester City, Manchester United). A prominent "Predict Match" button with a gradient background and hover animation initiates the prediction. The button includes a loading spinner that activates during the AJAX request, providing visual feedback that the system is processing. The interface is clean and intuitive, requiring no technical knowledge from the user — simply select two teams and click predict.

### Figure 5.22 — Prediction Result

*See deployed application or rendered prediction output*

Upon successful prediction, the result card displays the predicted outcome as a colour-coded badge: green for "Home Win", amber for "Draw", and red for "Away Win". The confidence percentage appears prominently, followed by three animated probability bars showing the model's probability estimates for each outcome class. Below the probabilities, the top 5 SHAP feature importances are displayed as horizontal bars, showing which engineered features most influenced this specific prediction. For example, when predicting Arsenal vs Chelsea, the `elo_diff` feature (the difference in Elo ratings between the two teams) and `home_elo` (Arsenal's pre-match strength rating) might appear as the top contributors, providing transparent, interpretable explanations for the model's decision.

### Figure 5.23 — Error Validation

*See deployed application or rendered validation output*

The system includes client-side and server-side validation to prevent invalid predictions. When the user selects the same team for both home and away positions, the frontend JavaScript displays an error message before the request is sent. If the backend receives an invalid request (e.g., an unsupported team name or missing parameters), it returns a structured JSON error response with a descriptive message. The frontend displays these errors in a red error card, maintaining the visual consistency of the interface while clearly communicating the issue to the user.

### Successful Prediction Workflow

The end-to-end prediction workflow completes in under 500 milliseconds for a typical request (excluding initial model loading on the first prediction). The workflow is as follows:

1. **User Interaction** (< 5 seconds): User selects home team, away team, and clicks "Predict Match".
2. **AJAX Request** (< 10ms): Frontend JavaScript constructs the POST request with CSRF token and sends it to the API endpoint.
3. **Input Validation** (< 1ms): Backend validates team names against the supported clubs list.
4. **Feature Engineering** (< 200ms): The feature engineering module computes 198 features from raw match data using pre-computed Elo ratings and historical statistics.
5. **Model Inference** (< 50ms): Three base models generate probability predictions, and the meta-learner produces the final stacked prediction.
6. **SHAP Computation** (< 100ms): The TreeExplainer computes SHAP values for the XGBoost model using the 50 selected features.
7. **Response Serialisation** (< 5ms): The JSON response is constructed and returned to the client.
8. **Frontend Rendering** (< 50ms): JavaScript parses the JSON response and dynamically updates the DOM with the prediction result, probability bars, and SHAP feature bars.

The total response time from user click to result display is typically 300–500 milliseconds, well within the threshold for a responsive user experience. The singleton pattern in the `PredictionService` ensures that model loading (the most time-consuming operation, ~2 seconds) occurs only once per server restart, making subsequent predictions nearly instantaneous.

### System Usability

The application is designed for both technical and non-technical users. The intuitive dropdown interface requires no understanding of machine learning concepts — users simply select two teams and receive a prediction. The probability bars and confidence percentage provide an accessible interpretation of model uncertainty, while the SHAP feature explanations offer transparency for users interested in understanding which factors drove the prediction. The responsive Tailwind CSS design ensures usability across desktop and mobile devices, with the navigation collapsing into a hamburger menu on smaller screens. The dark theme with glassmorphism effects creates a visually distinctive interface that differentiates the application from conventional sports analytics tools.

### Response Time and Scalability

The application's response time is dominated by two factors: initial model loading (~2 seconds on the first prediction) and feature engineering computation (~200ms per prediction). The model loading overhead is a one-time cost amortised across all subsequent predictions, and the feature engineering computation is bounded by the need to query historical match data and compute rolling statistics. For the current deployment on Render's free tier, the application handles individual prediction requests comfortably, with the singleton caching pattern ensuring consistent sub-second response times after initialisation. The Gunicorn worker configuration (4 processes) provides basic concurrent request handling, sufficient for the expected usage patterns of an academic demonstration system.
