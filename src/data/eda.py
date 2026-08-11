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

    # Analyzing missing values
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