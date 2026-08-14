# IMPORTS
import pandas as pd

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger
from src.utils.io_helpers import save_model, save_preprocessor
from src.data.loader import load_raw_data
from src.data.eda import EDA, get_summart_stats
from src.data.preprocessor import Preprocessor
from src.models.custom_logistic import CustomLogisticRegression
from src.training.trainer import train_custom_model
from src.evaluation.benchmark import run_benchmark
from src.evaluation.visualizer import generate_all_plots
from src.evaluation.metrics import print_metrics_comparison

# set up logger
logger = setup_logger(__name__)


def main():
    logger.info("\n")
    logger.info("starting churn prediction pipeline")
    logger.info("\n")

  
    # load raw data
    logger.info("loading raw data...")
    df = load_raw_data()
    logger.info("loaded %d rows and %d columns", df.shape[0], df.shape[1])

  
    # run exploratory data analysis
    logger.info("running exploratory data analysis...")
    eda_findings = EDA(df)
    logger.info("eda complete! yay!")
    logger.info("key findings: churn percentage = %.2f%%", eda_findings.get('churn_percentage', 0))

    # print summary statistics for reference
    summary_stats = get_summart_stats(df)
    logger.info("summary statistics:\n%s", summary_stats.to_string())

  
    # preprocess the data
    logger.info("preprocessing data...")
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    logger.info("preprocessing complete! wooh!")
    logger.info("training set: %d samples", X_train.shape[0])
    logger.info("test set: %d samples", X_test.shape[0])
    logger.info("features after encoding: %d", X_train.shape[1])

    # save the preprocessor for deployment
    preprocessor.save()
    logger.info("preprocessor saved successfully!")

  
    # train custom model
    logger.info("training custom model...")
    custom_model, custom_metrics = train_custom_model(X_train, X_test, y_train, y_test)

    logger.info("custom model training complete! success!")
    logger.info("custom model metrics:")
    logger.info("    accuracy: %.4f", custom_metrics['accuracy'])
    logger.info("    precision: %.4f", custom_metrics['precision'])
    logger.info("    recall: %.4f", custom_metrics['recall'])
    logger.info("    f1 score: %.4f", custom_metrics['f1'])
    logger.info("    roc-auc: %.4f", custom_metrics['roc_auc'])

  
    # benchmark against scikit-learn
    logger.info("running benchmark comparison...")
    sklearn_metrics = run_benchmark(X_train, X_test, y_train, y_test, custom_metrics)

  
    # generate all visualizations
    logger.info("generating all visualizations...")
    
    # need sklearn model for comparison plot
    from src.utils.io_helpers import load_model
    sklearn_model = load_model(Config.SKLEARN_MODEL_PATH)
    
    generate_all_plots(
        df=df,
        target_col=Config.TARGET_COL,
        custom_model=custom_model,
        sklearn_model=sklearn_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=preprocessor.feature_names,
        loss_history=custom_model.loss_history
    )
    
    logger.info("all visualizations generated! yay!")

  
    # final summary
    logger.info("\n")
    logger.info("pipeline complete! success!")
    logger.info("\n")
    
    # print final comparison summary
    print_metrics_comparison(custom_metrics, sklearn_metrics)

    logger.info("\n")
    logger.info("summary of saved artifacts:")
    logger.info("    custom model: %s", Config.CUSTOM_MODEL_PATH)
    logger.info("    sklearn model: %s", Config.SKLEARN_MODEL_PATH)
    logger.info("    preprocessor: %s", Config.PREPROCESSOR_PATH)
    logger.info("    plots: %s", Config.FIGURES_DIR)
    logger.info("    log file: %s", Config.LOG_PATH)
    
    return custom_model, custom_metrics, sklearn_metrics


if __name__ == "__main__":
    main()