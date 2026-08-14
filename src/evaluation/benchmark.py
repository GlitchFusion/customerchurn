# IMPORTS
import joblib

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_all_metrics, print_metrics_comparison
from src.training.trainer import train_sklearn_model

logger = setup_logger(__name__)


def run_benchmark(X_train, X_test, y_train, y_test, custom_metrics):
    logger.info("=" * 50)
    logger.info("starting benchmark comparison")
    logger.info("=" * 50)

    # [1] train sklearn model using the trainer function
    logger.info("[1] training scikit-learn model...")
    sklearn_metrics = train_sklearn_model(X_train, X_test, y_train, y_test)

    # [2] print side-by-side comparison
    logger.info("[2] comparison results:")
    print_metrics_comparison(custom_metrics, sklearn_metrics)

    # [3] determine which model performed better
    logger.info("[3] summary:")

    # compare f1 score (good for imbalanced data)
    custom_f1 = custom_metrics.get('f1', 0)
    sklearn_f1 = sklearn_metrics.get('f1', 0)

    if custom_f1 > sklearn_f1:
        logger.info("    custom model performed better on f1 score (%.4f vs %.4f) yay!",
                   custom_f1, sklearn_f1)
    elif custom_f1 < sklearn_f1:
        logger.info("    sklearn model performed better on f1 score (%.4f vs %.4f)",
                   sklearn_f1, custom_f1)
    else:
        logger.info("    both models have the same f1 score (%.4f)", custom_f1)

    # compare roc-auc
    custom_auc = custom_metrics.get('roc_auc', 0)
    sklearn_auc = sklearn_metrics.get('roc_auc', 0)

    if custom_auc > sklearn_auc:
        logger.info("    custom model performed better on roc-auc (%.4f vs %.4f) wooh!",
                   custom_auc, sklearn_auc)
    elif custom_auc < sklearn_auc:
        logger.info("    sklearn model performed better on roc-auc (%.4f vs %.4f)",
                   sklearn_auc, custom_auc)
    else:
        logger.info("    both models have the same roc-auc (%.4f)", custom_auc)

    logger.info("=" * 50)
    logger.info("benchmark complete! success!")
    logger.info("=" * 50)

    return sklearn_metrics


def compare_models(custom_metrics, sklearn_metrics):
    comparison = {}

    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for name in metric_names:
        custom_value = custom_metrics.get(name, 0)
        sklearn_value = sklearn_metrics.get(name, 0)
        difference = custom_value - sklearn_value

        comparison[name] = {
            'custom': custom_value,
            'sklearn': sklearn_value,
            'difference': difference,
            'better': 'custom' if difference > 0 else 'sklearn' if difference < 0 else 'tie'
        }

    return comparison