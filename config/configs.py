"""
configs.py - centralised configuration for the churn prediction project.

all paths, hyperparameters, and settings are defined here.
purpose: single source of truth for the entire pipeline.
"""

# IMPORTS
import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    configuration class for the churn prediction pipeline.
    contains all paths, parameters, and settings.
    """

    # 1. file and directory paths
    # root directory (go up one level from config folder)
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # data paths
    RAW_DATA_PATH: str = os.path.join(ROOT_DIR, "data/raw/Telco-Customer-Churn-Data.csv")
    PROCESSED_DATA_PATH: str = os.path.join(ROOT_DIR, "data/processed/churn_processed.csv")

    # model paths
    MODEL_DIR: str = os.path.join(ROOT_DIR, "models")
    CUSTOM_MODEL_PATH: str = os.path.join(MODEL_DIR, "custom_model.pkl")
    SKLEARN_MODEL_PATH: str = os.path.join(MODEL_DIR, "sklearn_model.pkl")
    PREPROCESSOR_PATH: str = os.path.join(MODEL_DIR, "preprocessor.pkl")

    # log path
    LOG_PATH: str = os.path.join(ROOT_DIR, "logs/training.log")

    # figures path
    FIGURES_DIR: str = os.path.join(ROOT_DIR, "reports/figures")

    # 2. data preprocessing settings
    TARGET_COL: str = "Churn"
    ID_COL: str = "customerID"

    NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

    CATEGORICAL_FEATURES = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # 3. data split parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # 4. model hyperparameters
    LEARNING_RATE: float = 0.01
    EPOCHS: int = 1000
    BATCH_SIZE: int = None  # none = full batch gradient descent

    # 5. regularization (handles overfitting)
    REGULARIZATION: str = None  # options: None, 'l1', 'l2'
    REG_LAMBDA: float = 0.1

    # 6. class imbalance (handles imbalanced data)
    CLASS_WEIGHT: str = 'balanced'  # options: 'balanced', None, or dict

    # 7. feature engineering parameters
    TENURE_BINS = [-1, 12, 24, 48, 100]
    TENURE_LABELS = ['0-12', '13-24', '25-48', '49+']

    # 8. deployment settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000