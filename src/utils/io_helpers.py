"""
io_helpers.py - helper functions for saving and loading models.

this module provides functions to:
- save models and preprocessors to disk
- load models and preprocessors from disk
- check if files exist

purpose: to keep file i/o operations in one place for reusability.
"""

# IMPORTS
import os
import joblib

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger

# at the top of io_helpers.py
logger = setup_logger(__name__)
if logger is None:
    # fallback: create a basic logger
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("fallback logger used - no file logging available")


def save_model(model, filepath):
    """
    save a model to disk using joblib.

    arguments:
        model: trained model object (custom or sklearn)
        filepath: path where the model should be saved

    returns:
        bool: true if save was successful, false otherwise
    """
    try:
        # check if directory exists, create if not
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("created directory: %s", directory)

        # save the model
        joblib.dump(model, filepath)
        logger.info("model saved successfully to: %s yay!", filepath)
        return True

    except Exception as error:
        logger.error("failed to save model to %s: %s", filepath, str(error))
        return False


def load_model(filepath):
    """
    load a model from disk using joblib.

    arguments:
        filepath: path to the saved model file

    returns:
        object: loaded model, or none if loading fails
    """
    try:
        if not os.path.exists(filepath):
            logger.error("model file not found: %s", filepath)
            return None

        model = joblib.load(filepath)
        logger.info("model loaded successfully from: %s wooh!", filepath)
        return model

    except Exception as error:
        logger.error("failed to load model from %s: %s", filepath, str(error))
        return None


def save_preprocessor(preprocessor, filepath=Config.PREPROCESSOR_PATH):
    """
    save the fitted preprocessor to disk.

    this is a convenience wrapper around save_model().

    arguments:
        preprocessor: fitted preprocessor object
        filepath: path where the preprocessor should be saved

    returns:
        bool: true if save was successful, false otherwise
    """
    try:
        # preprocessor should have a save method
        preprocessor.save(filepath)
        return True

    except Exception as error:
        logger.error("failed to save preprocessor: %s", str(error))
        return False


def load_preprocessor(filepath=Config.PREPROCESSOR_PATH):
    """
    load the saved preprocessor from disk.

    arguments:
        filepath: path to the saved preprocessor file

    returns:
        object: loaded preprocessor, or none if loading fails
    """
    try:
        if not os.path.exists(filepath):
            logger.error("preprocessor file not found: %s", filepath)
            return None

        data = joblib.load(filepath)
        logger.info("preprocessor loaded successfully from: %s wooh!", filepath)
        return data

    except Exception as error:
        logger.error("failed to load preprocessor: %s", str(error))
        return None


def check_file_exists(filepath):
    """
    check if a file exists at the given path.

    arguments:
        filepath: path to check

    returns:
        bool: true if file exists, false otherwise
    """
    exists = os.path.exists(filepath)
    if exists:
        logger.info("file exists: %s", filepath)
    else:
        logger.warning("file does not exist: %s", filepath)
    return exists


def get_model_paths():
    """
    get all model paths from config.

    returns:
        dict: dictionary with model paths
    """
    paths = {
        'custom_model': Config.CUSTOM_MODEL_PATH,
        'sklearn_model': Config.SKLEARN_MODEL_PATH,
        'preprocessor': Config.PREPROCESSOR_PATH
    }
    return paths