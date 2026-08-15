# importing modules
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# importing local modules
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_all_metrics(y_true, y_pred, y_pred_proba):
    # calculate each metric
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # getting confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    logger.info("confusion matrix:")
    logger.info("    true negatives: %d", tn)
    logger.info("    false positives: %d", fp)
    logger.info("    false negatives: %d", fn)
    logger.info("    true positives: %d", tp)

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
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("\n" + "=" * 60)
    print("performance comparison: custom vs sklearn")
    print("=" * 60)

    print(f"{'metric':<15} {'custom':<15} {'sklearn':<15} {'difference':<15}")
    print("-" * 60)

    for name in metric_names:
        custom_value = custom_metrics.get(name, 0)
        sklearn_value = sklearn_metrics.get(name, 0)
        difference = custom_value - sklearn_value
        print(f"{name:<15} {custom_value:<15.4f} {sklearn_value:<15.4f} {difference:+.4f}")

    print("=" * 60)