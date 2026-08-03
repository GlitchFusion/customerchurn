import os
from dataclasses import dataclass

@dataclass
class Config:
      # paths to every required directory - just in case

      # Root directory of the project
      ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

      # Data paths
      DATA_DIR:str = os.path.join(ROOT_DIR, "data/raw/Telco-Customer-Churn-Data.csv")
      PROCESSED_DATA_DIR:str = os.path.join(ROOT_DIR, "data/processed")

      # Model paths
      MODEL_DIR:str = os.path.join(ROOT_DIR, "models")


      # DATA PREPROCESSING PARAMETERS
      # Target column name
      TARGET_COL:str = "Churn"
      ID_COL:str = "customerID"

      # Features (based on Telco Customer Churn dataset)
      NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
      CATEGORICAL_FEATURES = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService' 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod' ]