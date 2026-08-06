import os
from dataclasses import dataclass

@dataclass # Decorator to automatically generate init, repr, and other methods
class Config:
      # paths to every required directory - just in case

      # Root directory of the project
      ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

      # Data paths
      RAW_DATA_DIR:str = os.path.join(ROOT_DIR, "data/raw/Telco-Customer-Churn-Data.csv")

      # Model paths
      MODEL_DIR:str = os.path.join(ROOT_DIR, "models")

      # Logs paths
      LOGS_DIR:str = os.path.join(ROOT_DIR, "logs/training.log")

      # DATA PREPROCESSING PARAMETERS
      # Target column name
      TARGET_COL:str = "Churn"
      ID_COL:str = "customerID"

      # Features
      NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
      CATEGORICAL_FEATURES = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod' ]

      # PARAMETERS: Data
      TEST_SIZE: float = 0.2
      TRAIN_SIZE: float = 0.8
      RANDOM_STATE: int = 42

      # HYPERPARAMETERS: Model
      N_ESTIMATORS: int = 100
      LEARNING_RATE: float = 0.1
      BATCH_SIZE: int = None

      # REGULARIZATION: Handeling Overfitting
      REGULARIZATION: str = None
      REG_LAMBDA: float = 0.01

      # CLASS IMBALANCE: Handling Class Imbalance
      CLASS_WEIGHT: str = None

      # FEATURE ENGINEERING: Feature Engineering Parameters
      BINS = [-1, 12, 24, 48, 100]
      LABELS = ['0-12', '13-24', '24-48', '49+']