# IMPORTS
import joblib
from sklearn.linear_model import LogisticRegression

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger
from src.models.custom_logistic import CustomLogisticRegression
from src.evaluation.metrics import compute_all_metrics

logger = setup_logger(__name__)


def train_custom_model(X_train, X_test, y_train, y_test):
    logger.info("starting custom model training")


    # [1] initialize the custom model with parameters from config

    model = CustomLogisticRegression(
        learning_rate=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        class_weight=Config.CLASS_WEIGHT,
        reg=Config.REGULARIZATION,
        reg_lambda=Config.REG_LAMBDA
    )

    logger.info("[1] model initialized with:")
    logger.info("    learning_rate: %s", Config.LEARNING_RATE)
    logger.info("    epochs: %s", Config.EPOCHS)
    logger.info("    class_weight: %s", Config.CLASS_WEIGHT)
    logger.info("    regularization: %s", Config.REGULARIZATION)


    # [2] train the model

    logger.info("[2] training model...")
    model.fit(X_train, y_train)


    # [3] get predictions on test data

    logger.info("[3] generating predictions on test data...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    logger.info("    predictions complete")


    # [4] compute evaluation metrics

    logger.info("[4] computing evaluation metrics...")
    metrics = compute_all_metrics(y_test, y_pred, y_pred_proba)

    logger.info("    accuracy: %.4f", metrics['accuracy'])
    logger.info("    precision: %.4f", metrics['precision'])
    logger.info("    recall: %.4f", metrics['recall'])
    logger.info("    f1 score: %.4f", metrics['f1'])
    logger.info("    roc_auc: %.4f", metrics['roc_auc'])


    # [5] save the trained model to disk

    logger.info("[5] saving model to disk...")
    joblib.dump(model, Config.CUSTOM_MODEL_PATH)
    logger.info("    model saved to: %s", Config.CUSTOM_MODEL_PATH)


    # [6] log final loss for reference

    final_loss = model.loss_history[-1] if model.loss_history else None
    logger.info("[6] final training loss: %.6f", final_loss)

    logger.info("=" * 50)
    logger.info("training complete!")
    logger.info("=" * 50)

    return model, metrics


def train_sklearn_model(X_train, X_test, y_train, y_test):
    logger.info("=" * 50)
    logger.info("training scikit-learn model for benchmarking")
    logger.info("=" * 50)


    # [1] initialize sklearn model with similar parameters
    model = LogisticRegression(
        class_weight=Config.CLASS_WEIGHT,
        random_state=Config.RANDOM_STATE,
        max_iter=1000,
        solver='lbfgs'
    )

    logger.info("[1] sklearn model initialized")


    # [2] train the model
    logger.info("[2] training sklearn model...")
    model.fit(X_train, y_train)


    # [3] get predictions
    logger.info("[3] generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]


    # [4] compute metrics
    logger.info("[4] computing evaluation metrics...")
    metrics = compute_all_metrics(y_test, y_pred, y_pred_proba)

    logger.info("    accuracy: %.4f", metrics['accuracy'])
    logger.info("    precision: %.4f", metrics['precision'])
    logger.info("    recall: %.4f", metrics['recall'])
    logger.info("    f1 score: %.4f", metrics['f1'])
    logger.info("    roc_auc: %.4f", metrics['roc_auc'])


    # [5] save the sklearn model
    logger.info("[5] saving sklearn model...")
    joblib.dump(model, Config.SKLEARN_MODEL_PATH)
    logger.info("    model saved to: %s", Config.SKLEARN_MODEL_PATH)

    logger.info("=" * 50)
    logger.info("sklearn training complete!")
    logger.info("=" * 50)

    return metrics