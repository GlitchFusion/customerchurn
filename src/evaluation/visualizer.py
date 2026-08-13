"""
visualizer.py - centralized plotting hub for all visualizations.

contains every plot type needed for the project:
- eda (distributions, correlations, churn rates)
- training diagnostics (loss curve)
- model evaluation (confusion matrix, roc, pr curves)
- model interpretation (feature importance, threshold tuning)

purpose: to provide reusable plotting functions that save
to reports/figures/ or display interactively.
"""

# IMPORTS
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, precision_score,
    recall_score, f1_score
)

# LOCAL IMPORTS
from config.configs import Config

# styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


def _ensure_fig_dir():
    """ensure the figures directory exists."""
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)


def _save_or_show(fig, filename=None, save=True):
    """
    helper to save to reports/figures/ or display.

    arguments:
        fig: matplotlib figure object
        filename: name of the file (without extension)
        save: if true, save to disk; if false, display.
    """
    if save and filename:
        _ensure_fig_dir()
        filepath = os.path.join(Config.FIGURES_DIR, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"saved: {filepath}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# exploratory data analysis (eda) plots
# ----------------------------------------------------------------------

def plot_target_distribution(df, target_col='Churn', save=True):
    """
    bar chart of target variable (churn vs non-churn).

    arguments:
        df: dataset
        target_col: name of target column
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target_col].value_counts().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('target variable distribution', fontsize=14)
    ax.set_xlabel('churn')
    ax.set_ylabel('count')
    ax.set_xticklabels(['not churned', 'churned'], rotation=0)
    for p in ax.patches:
        ax.annotate(str(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    _save_or_show(fig, 'target_distribution', save)


def plot_categorical_churn(df, cat_col, target_col='Churn', save=True):
    """
    stacked bar chart showing churn rate by a categorical feature.

    arguments:
        df: dataset
        cat_col: categorical column name
        target_col: target column name
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_rate = pd.crosstab(df[cat_col], df[target_col], normalize='index') * 100
    churn_rate.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title(f'churn rate by {cat_col}', fontsize=14)
    ax.set_ylabel('percentage (%)')
    ax.set_xlabel(cat_col)
    ax.legend(['not churned', 'churned'])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    _save_or_show(fig, f'churn_by_{cat_col}', save)


def plot_numerical_distribution(df, num_col, save=True):
    """
    histogram + kde for a numerical feature.

    arguments:
        df: dataset
        num_col: numerical column name
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[num_col], kde=True, ax=ax, color='#2ca02c')
    ax.set_title(f'distribution of {num_col}', fontsize=14)
    ax.set_xlabel(num_col)
    _save_or_show(fig, f'distribution_{num_col}', save)


def plot_correlation_matrix(df, numeric_cols=None, save=True):
    """
    heatmap of feature correlations.

    arguments:
        df: dataset (numeric columns only)
        numeric_cols: list of numeric columns (optional)
        save: save to disk or display
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        print("not enough numeric columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('feature correlation matrix', fontsize=14)
    _save_or_show(fig, 'correlation_heatmap', save)


def plot_churn_boxplot(df, num_col, target_col='Churn', save=True):
    """
    box plot of a numerical feature segmented by churn.

    arguments:
        df: dataset
        num_col: numerical column name
        target_col: target column name
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=target_col, y=num_col, data=df, ax=ax,
                palette=['#1f77b4', '#ff7f0e'])
    ax.set_title(f'{num_col} by churn', fontsize=14)
    _save_or_show(fig, f'boxplot_{num_col}_by_churn', save)


# ----------------------------------------------------------------------
# training diagnostics
# ----------------------------------------------------------------------

def plot_loss_curve(loss_history, save=True):
    """
    plot log-loss over training epochs.

    arguments:
        loss_history: list of loss values per epoch
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, color='#d62728', linewidth=2)
    ax.set_title('loss over epochs (training)', fontsize=14)
    ax.set_xlabel('epochs')
    ax.set_ylabel('log-loss (cross-entropy)')
    ax.grid(alpha=0.3)
    _save_or_show(fig, 'loss_curve', save)


# ----------------------------------------------------------------------
# model evaluation plots
# ----------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels=None, save=True):
    """
    confusion matrix with annotations.

    labels are removed from the plot and will be shown on the webpage.
    this gives cleaner visualizations without clutter.

    arguments:
        y_true: true labels
        y_pred: predicted labels
        labels: not used - kept for compatibility (labels shown on webpage)
        save: save to disk or display
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # create heatmap without tick labels (they will be shown on webpage)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=False, yticklabels=False, cbar=True)
    
    ax.set_title('confusion matrix', fontsize=14)
    ax.set_xlabel('predicted', fontsize=12)
    ax.set_ylabel('actual', fontsize=12)
    
    _save_or_show(fig, 'confusion_matrix', save)


def plot_roc_curve(y_true, y_proba, model_name='custom model', save=True):
    """
    roc curve with auc score.

    arguments:
        y_true: true labels
        y_proba: predicted probabilities
        model_name: name for legend
        save: save to disk or display

    returns:
        float: auc score
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (auc = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('false positive rate', fontsize=12)
    ax.set_ylabel('true positive rate', fontsize=12)
    ax.set_title('roc curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    _save_or_show(fig, 'roc_curve', save)
    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, model_name='custom model', save=True):
    """
    precision-recall curve (useful for imbalanced data).

    arguments:
        y_true: true labels
        y_proba: predicted probabilities
        model_name: name for legend
        save: save to disk or display
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, linewidth=2, label=model_name)
    ax.set_xlabel('recall', fontsize=12)
    ax.set_ylabel('precision', fontsize=12)
    ax.set_title('precision-recall curve', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    _save_or_show(fig, 'precision_recall_curve', save)


def plot_roc_comparison(y_true, custom_proba, sklearn_proba, save=True):
    """
    compare roc curves of custom vs sklearn model on one plot.

    arguments:
        y_true: true labels
        custom_proba: custom model probabilities
        sklearn_proba: sklearn model probabilities
        save: save to disk or display
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for proba, name in [(custom_proba, 'custom model'), (sklearn_proba, 'scikit-learn')]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (auc = {roc_auc:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('false positive rate', fontsize=12)
    ax.set_ylabel('true positive rate', fontsize=12)
    ax.set_title('roc curve comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    _save_or_show(fig, 'roc_comparison', save)


# ----------------------------------------------------------------------
# model interpretation plots
# ----------------------------------------------------------------------

def plot_feature_importance(weights, feature_names, top_n=20, save=True):
    """
    bar chart of logistic regression coefficients.
    positive = increases churn risk, negative = decreases churn risk.

    arguments:
        weights: model coefficients/weights
        feature_names: names of features
        top_n: number of top features to display
        save: save to disk or display
    """
    coef_df = pd.DataFrame({'feature': feature_names, 'weight': weights})
    coef_df['abs_weight'] = coef_df['weight'].abs()
    coef_df = coef_df.sort_values('abs_weight', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ['#ff7f0e' if w > 0 else '#1f77b4' for w in coef_df['weight']]
    ax.barh(coef_df['feature'], coef_df['weight'], color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'top {top_n} feature coefficients (impact on churn)', fontsize=14)
    ax.set_xlabel('coefficient weight')
    ax.set_ylabel('features')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    _save_or_show(fig, 'feature_importance', save)


def plot_threshold_analysis(y_true, y_proba, save=True):
    """
    plot precision, recall, and f1 across different decision thresholds.
    helps choose the optimal threshold for business needs.

    arguments:
        y_true: true labels
        y_proba: predicted probabilities
        save: save to disk or display
    """
    thresholds = np.arange(0.0, 1.0, 0.02)
    precisions = []
    recalls = []
    f1s = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label='precision', linewidth=2)
    ax.plot(thresholds, recalls, label='recall', linewidth=2)
    ax.plot(thresholds, f1s, label='f1 score', linewidth=2, linestyle='--')
    ax.set_xlabel('threshold')
    ax.set_ylabel('score')
    ax.set_title('precision, recall, and f1 vs. threshold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, 'threshold_analysis', save)


# ----------------------------------------------------------------------
# batch generator – run everything at once!
# ----------------------------------------------------------------------

def generate_all_plots(df, target_col, custom_model, sklearn_model,
                       X_test, y_test, feature_names, loss_history):
    """
    master function that generates every plot in one call.
    call this from main.py to produce all figures for your report.

    arguments:
        df: raw dataset
        target_col: target column name
        custom_model: fitted custom logistic regression model
        sklearn_model: fitted sklearn model
        X_test: test features
        y_test: test labels
        feature_names: names of features
        loss_history: loss history from training
    """
    print("generating all plots...")

    # eda plots
    plot_target_distribution(df, target_col, save=True)
    plot_correlation_matrix(df, save=True)
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            plot_numerical_distribution(df, col, save=True)
    for col in ['Contract', 'InternetService', 'PaymentMethod']:
        if col in df.columns:
            plot_categorical_churn(df, col, target_col, save=True)

    # predictions
    y_pred_custom = custom_model.predict(X_test)
    y_proba_custom = custom_model.predict_proba(X_test)
    y_proba_sklearn = sklearn_model.predict_proba(X_test)[:, 1]

    # evaluation plots
    plot_confusion_matrix(y_test, y_pred_custom, save=True)
    plot_roc_comparison(y_test, y_proba_custom, y_proba_sklearn, save=True)
    plot_precision_recall_curve(y_test, y_proba_custom, save=True)
    plot_roc_curve(y_test, y_proba_custom, model_name='custom model', save=True)

    # training diagnostics
    plot_loss_curve(loss_history, save=True)

    # feature interpretation
    plot_feature_importance(custom_model.weights, feature_names, top_n=15, save=True)

    # threshold tuning
    plot_threshold_analysis(y_test, y_proba_custom, save=True)

    print("all plots saved to reports/figures/")