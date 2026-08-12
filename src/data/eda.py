"""
eda.py - Contains functions for exploatory data analysis on the provided dataset.
Generate text summaries, statistical insights, and visualizations

Purpose of this module: To understand the raw data before preprocessing 
"""

# IMPROTS
import pandas as pd
import numpy as np

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger
from src.evaluation.visualizer import (
    plot_target_distribution,
    plot_categorical_churn,
    plot_numerical_distribution,
    plot_correlation_matrix,
    plot_churn_boxplot,
)

logger = setup_logger(__name__)

def EDA(df: pd.DataFrame) -> dict:
    """
    Objective: performing EDA on raw dataset
    Arguments:
        df: pd.DataFrame -> This is the raw dataset loaded from the loader.py
    
    Returns? A dictornary that contains EDA findings and insights. 
    """

    logger.info("COMMENCING EDA ANALYSIS")

    findings = {}

    # BASIC INFORMATION
    logger.info("[1] Basic information about the dataset")
    logger.info(f"Dataset shape: {df.shape}\nRows: {df.shape[0]}\nColumns: {df.shape[1]}")
    logger.info(f"Data Types: \n{df.dtypes}")

    # ANALYZING MISSING VALUES
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    missing_values_df = pd.DataFrame({
        "Missing count": missing_values,
        "Missing Percentge": missing_values_percentage
    })
    missing_values_df = missing_values_df[missing_values_df["Missing count"] > 0]

    if not missing_values_df.empty:
        logger.info(f"[2] missing values\n {missing_values_df}")
        findings["missing_values"] = missing_values_df.dict()
    else:
        logger.info("[2] No missing values found!")
        findings["missing_values"] = {}

    # TARGET VARAIABLE DISTIBUTION
    logger.info("[3] Target variable analysis")
    target = Config.TARGET_COL
    if target in df.columns:
        churn_count = df[target].value_counts()
        churn_percentage = (churn_count / len(df)) * 100
        logger.info(f"Churn Distribution: \n{churn_count}\nPercentage: {churn_percentage}")
        findings["churn_percentage"] = churn_percentage.get("yes", "0")

        # generating the target distribution plot
        plot_target_distribution(df, target_col=target, save=True)
    else:
        logger.warning(f"target column '{target}' not found in dataset")

    # CATEGORICAL FEATURE ANALYSIS
    logger.info("[4] Categorical Feature Analysis")
    category_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # excluding target and id columns
    exclude = [Config.TARGET_COL, Config.ID_COL]
    category_cols = [c for c in category_cols if c not in exclude]

    for col in category_cols:
        value_counts = df[col].value_counts()
        logger.info(f"- {col}: {len(value_counts)} unique values")
        logger.info(f"   Top 3: {value_counts.head(3).to_dict()}")

        # generating churn rate plot for this categorical feature
        if target in df.columns:
            plot_categorical_churn(df, col, target_col = target, save=True)

    # NUMERIC FEATURE ANAYZIN
    logger.info("[5] Numeric Feature Analysis")