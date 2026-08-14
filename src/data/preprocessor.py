# IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Preprocessor:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit_transform(self, df: pd.DataFrame):
        logger.info("starting preprocessing pipeline...")


        # [1] drop customer id column
        if Config.ID_COL in df.columns:
            df = df.drop(columns=[Config.ID_COL])
            logger.info("[1] dropped '%s' column", Config.ID_COL)


        # [2] fix totalcharges
        if 'TotalCharges' in df.columns:
            # convert to numeric (empty strings become nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # fill nans with median (very few missing values)
            median_val = df['TotalCharges'].median()
            df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
            logger.info("[2] fixed totalcharges: filled missing with median (%.2f)", median_val)


        # [3] create new features
        # 3a. tenure_group (categorical bins)
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=Config.TENURE_BINS,
            labels=Config.TENURE_LABELS
        )
        logger.info("[3a] created tenure_group with bins: %s", Config.TENURE_LABELS)

        # 3b. avg_monthly_spend = totalcharges / (tenure + 1)
        # add 1 to avoid division by zero for tenure = 0
        df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
        logger.info("[3b] created avg_monthly_spend feature")


        # [4] separate features and target

        target = Config.TARGET_COL
        if target not in df.columns:
            raise ValueError(f"target column '{target}' not found in dataframe")

        X = df.drop(columns=[target])
        y = df[target].apply(lambda x: 1 if x == 'Yes' else 0).values
        logger.info("[4] target distribution: %d churned, %d not churned",
                   sum(y), len(y) - sum(y))


        # [5] define column types for transformer
        # numeric features will be scaled
        numeric_features = [
            'tenure',
            'MonthlyCharges',
            'TotalCharges',
            'avg_monthly_spend'
        ]

        # categorical features will be one-hot encoded
        # seniorcitizen is numeric (0/1), not categorical in this dataset
        categorical_features = [
            'gender', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure_group'
        ]

        # filter to only include columns that actually exist
        numeric_features = [col for col in numeric_features if col in X.columns]
        categorical_features = [col for col in categorical_features if col in X.columns]

        logger.info("[5] numeric features: %s", numeric_features)
        logger.info("[5] categorical features: %s", categorical_features)


        # [6] build columntransformer pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # drop any columns not explicitly handled
        )

        # fit and transform
        X_processed = preprocessor.fit_transform(X)
        logger.info("[6] preprocessor fitted. transformed shape: %s", X_processed.shape)

        # store pipeline for later use (deployment)
        self.pipeline = preprocessor

        # store feature names for interpretation
        num_feature_names = numeric_features

        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

        self.feature_names = num_feature_names + cat_feature_names
        logger.info("[6] total features after encoding: %d", len(self.feature_names))


        # [7] train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed,
            y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y  # preserve class distribution
        )

        logger.info("[7] train/test split: train %s, test %s",
                   self.X_train.shape, self.X_test.shape)

        logger.info("✅ preprocessing complete!")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def transform(self, X_raw: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("preprocessor has not been fitted. call fit_transform() first.")

        X = X_raw.copy()

        # create tenure_group
        X['tenure_group'] = pd.cut(
            X['tenure'],
            bins=Config.TENURE_BINS,
            labels=Config.TENURE_LABELS
        )

        # create avg_monthly_spend
        X['avg_monthly_spend'] = X['TotalCharges'] / (X['tenure'] + 1)

        X_processed = self.pipeline.transform(X)
        return X_processed

    def save(self, filepath=Config.PREPROCESSOR_PATH):
        if self.pipeline is None:
            raise ValueError("preprocessor has not been fitted. call fit_transform() first.")

        joblib.dump({
            'pipeline': self.pipeline,
            'feature_names': self.feature_names
        }, filepath)
        logger.info("WOOOOH!! preprocessor saved to: %s", filepath)

    def load(self, filepath=Config.PREPROCESSOR_PATH):
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.feature_names = data['feature_names']
        logger.info("WOOOOH!! preprocessor loaded from: %s", filepath)