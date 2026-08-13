"""
metrics.py - contains evaluation metrics for model performance.

this module provides functions to compute:
- accuracy
- precision
- recall
- f1 score
- roc-auc

purpose: to have a single place for all evaluation metrics.
"""

# IMPORTS
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# LOCAL IMPORTS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_all_metrics(y_true, y_pred, y_pred_proba):
    """
    compute all evaluation metrics at once.

    this function calculates:
    - accuracy: percentage of correct predictions
    - precision: how many predicted positives are actually positive
    - recall: how many actual positives were captured
    - f1 score: harmonic mean of precision and recall
    - roc-auc: area under the roc curve

    arguments:
        y_true: true labels (0 or 1)
        y_pred: predicted labels (0 or 1)
        y_pred_proba: predicted probabilities (between 0 and 1)

    returns:
        dict: dictionary containing all metrics
    """
    # calculate each metric
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # get confusion matrix for reference
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # log the confusion matrix values
    logger.info("confusion matrix:")
    logger.info("    true negatives: %d", tn)
    logger.info("    false positives: %d", fp)
    logger.info("    false negatives: %d", fn)
    logger.info("    true positives: %d", tp)

    # create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

    return metrics


def print_metrics_comparison(custom_metrics, sklearn_metrics):
    """
    print a side-by-side comparison of custom vs sklearn metrics.

    arguments:
        custom_metrics: metrics dictionary from custom model
        sklearn_metrics: metrics dictionary from sklearn model
    """
    # only compare the main metrics (not confusion matrix values)
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("\n" + "=" * 60)
    print("performance comparison: custom vs sklearn")
    print("=" * 60)

    # header
    print(f"{'metric':<15} {'custom':<15} {'sklearn':<15} {'difference':<15}")
    print("-" * 60)

    # each metric
    for name in metric_names:
        custom_value = custom_metrics.get(name, 0)
        sklearn_value = sklearn_metrics.get(name, 0)
        difference = custom_value - sklearn_value
        print(f"{name:<15} {custom_value:<15.4f} {sklearn_value:<15.4f} {difference:+.4f}")

    print("=" * 60)