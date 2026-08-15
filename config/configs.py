# importing modules
import os
from dataclasses import dataclass


@dataclass
class Config:

    # setting file and directory paths
    # setting root directory
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # setting data paths
    RAW_DATA_PATH: str = os.path.join(ROOT_DIR, "data/raw/Telco-Customer-Churn-Data.csv")
    PROCESSED_DATA_PATH: str = os.path.join(ROOT_DIR, "data/processed/churn_processed.csv")

    # setting model paths
    MODEL_DIR: str = os.path.join(ROOT_DIR, "models")
    CUSTOM_MODEL_PATH: str = os.path.join(MODEL_DIR, "custom_model.pkl")
    SKLEARN_MODEL_PATH: str = os.path.join(MODEL_DIR, "sklearn_model.pkl")
    PREPROCESSOR_PATH: str = os.path.join(MODEL_DIR, "preprocessor.pkl")

    # setting log path
    LOG_PATH: str = os.path.join(ROOT_DIR, "logs/training.log")

    # setting figures path
    FIGURES_DIR: str = os.path.join(ROOT_DIR, "reports/figures")

    # setting data preprocessing parameters
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

    # setting data split parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # setting model hyperparameters
    LEARNING_RATE: float = 0.01
    EPOCHS: int = 1000
    BATCH_SIZE: int = None

    # applying regularization
    REGULARIZATION: str = None
    REG_LAMBDA: float = 0.1

    # handling class imbalance
    CLASS_WEIGHT: str = 'balanced'

    # setting feature engineering parameters
    TENURE_BINS = [-1, 12, 24, 48, 100]
    TENURE_LABELS = ['0-12', '13-24', '25-48', '49+']

    # setting deployment parameters
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000