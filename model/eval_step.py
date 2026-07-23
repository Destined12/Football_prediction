import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'web', 'trained_models')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'model')

LABEL_NAMES = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
MODEL_COLORS = {'XGBoost': '#f97316', 'SVM': '#3b82f6', 'Random Forest': '#10b981', 'Ensemble': '#a855f7'}


def load_data():
    path = os.path.join(RESULTS_DIR, 'engineered_features', 'engineered_features.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    selected_path = os.path.join(MODELS_DIR, 'selected_features.pkl')
    selected_features = joblib.load(selected_path)
    return df, selected_features


def chronological_split(df, test_season='2024-2025'):
    train = df[df['season'] != test_season].copy()
    test = df[df['season'] == test_season].copy()
    return train, test


def load_all_models():
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_pipeline.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
    meta_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_meta_model.pkl'))
    return {'XGBoost': xgb_model, 'SVM': svm_model, 'Random Forest': rf_model}, meta_model


def evaluate_model(y_true, y_proba, model_name):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, balanced_accuracy_score, classification_report
    )
    y_pred = np.argmax(y_proba, axis=1)
    report = classification_report(y_true, y_pred, target_names=list(LABEL_NAMES.values()), output_dict=True)
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'),
        'log_loss_value': log_loss(y_true, y_proba),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    return metrics, y_pred, report


def plot_confusion_matrix(y_true, y_pred, model_name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=list(LABEL_NAMES.values()),
                yticklabels=list(LABEL_NAMES.values()),
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(' ', '_')
    fig.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{safe_name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{safe_name}.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_metric_comparison(all_metrics):
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Acc']
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    models = [m['model_name'] for m in all_metrics]
    colors = [MODEL_COLORS.get(m, '#94a3b8') for m in models]
    for ax, metric, label in zip(axes, metrics_to_plot, labels):
        vals = [m[metric] for m in all_metrics]
        bars = ax.bar(models, vals, color=colors, edgecolor='#0f172a')
        ax.set_title(label, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'metric_comparison.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'metric_comparison.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(y_true, all_probas, all_names):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    n_classes = 3
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = list(LABEL_NAMES.values())
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        for name, proba in zip(all_names, all_probas):
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], proba[:, class_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', color=MODEL_COLORS.get(name, '#94a3b8'), linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(f'ROC Curve — {class_name}', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('ROC Curves per Class', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'roc_curves.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_log_loss_comparison(all_metrics):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [m['model_name'] for m in all_metrics]
    losses = [m['log_loss_value'] for m in all_metrics]
    colors = [MODEL_COLORS.get(m, '#94a3b8') for m in models]
    bars = ax.bar(models, losses, color=colors, edgecolor='#0f172a')
    ax.set_title('Log Loss Comparison (Lower is Better)', fontweight='bold')
    ax.set_ylabel('Log Loss')
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'log_loss_comparison.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'log_loss_comparison.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_precision_recall_curves(y_true, y_proba):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    class_names = list(LABEL_NAMES.values())
    colors_class = ['#10b981', '#f59e0b', '#ef4444']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for class_idx, (ax, class_name, color) in enumerate(zip(axes, class_names, colors_class)):
        precision, recall, _ = precision_recall_curve(y_bin[:, class_idx], y_proba[:, class_idx])
        ap = average_precision_score(y_bin[:, class_idx], y_proba[:, class_idx])
        ax.plot(recall, precision, color=color, linewidth=2, label=f'Ensemble (AP={ap:.3f})')
        ax.fill_between(recall, precision, alpha=0.15, color=color)
        ax.set_title(f'Precision-Recall — {class_name}', fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Precision-Recall Curves — Stacking Ensemble', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'pr_curves.svg'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved pr_curves.png/svg', flush=True)


def run_evaluation():
    t_start = time.time()
    print('=' * 60, flush=True)
    print('MODEL EVALUATION (v2 — XGBoost + SVM + RF + Ensemble)', flush=True)
    print('=' * 60, flush=True)

    df, selected_features = load_data()
    train_df, test_df = chronological_split(df)
    X_test = test_df[selected_features].fillna(0).values
    y_test = test_df['ftr_encoded'].values.astype(int)
    print(f'Test set: {len(test_df)} matches, {len(selected_features)} features', flush=True)

    base_models, meta_model = load_all_models()
    print('All models loaded.', flush=True)

    all_metrics = []
    all_probas = []
    all_names = []

    for name, model in base_models.items():
        proba = model.predict_proba(X_test)
        metrics, y_pred, report = evaluate_model(y_test, proba, name)
        all_metrics.append(metrics)
        all_probas.append(proba)
        all_names.append(name)
        print(f'\n  {name}:', flush=True)
        print(f'    Accuracy:  {metrics["accuracy"]:.4f}', flush=True)
        print(f'    F1 Macro:  {metrics["f1_macro"]:.4f}', flush=True)
        print(f'    ROC-AUC:   {metrics["roc_auc"]:.4f}', flush=True)
        print(f'    Log Loss:  {metrics["log_loss_value"]:.4f}', flush=True)
        plot_confusion_matrix(y_test, y_pred, name)

    stack_test = np.concatenate([m.predict_proba(X_test) for m in base_models.values()], axis=1)
    stack_proba = meta_model.predict_proba(stack_test)
    metrics, y_pred, report = evaluate_model(y_test, stack_proba, 'Ensemble')
    all_metrics.append(metrics)
    all_probas.append(stack_proba)
    all_names.append('Ensemble')
    print(f'\n  Ensemble (Stacking):', flush=True)
    print(f'    Accuracy:  {metrics["accuracy"]:.4f}', flush=True)
    print(f'    F1 Macro:  {metrics["f1_macro"]:.4f}', flush=True)
    print(f'    ROC-AUC:   {metrics["roc_auc"]:.4f}', flush=True)
    print(f'    Log Loss:  {metrics["log_loss_value"]:.4f}', flush=True)
    plot_confusion_matrix(y_test, y_pred, 'Ensemble')

    print('\n--- Generating Comparison Plots ---', flush=True)
    plot_metric_comparison(all_metrics)
    plot_roc_curves(y_test, all_probas, all_names)
    plot_log_loss_comparison(all_metrics)
    plot_precision_recall_curves(y_test, stack_proba)

    print('\n--- Saving Metrics ---', flush=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'tables', 'evaluation_metrics.csv'), index=False)

    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()), output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(os.path.join(RESULTS_DIR, 'tables', 'classification_report.csv'))

    print(f'\n  Saved evaluation_metrics.csv', flush=True)
    print(f'  Saved classification_report.csv', flush=True)
    print(f'  All figures saved to {FIGURES_DIR}', flush=True)
    print(f'\n  Total time: {time.time()-t_start:.1f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    run_evaluation()
