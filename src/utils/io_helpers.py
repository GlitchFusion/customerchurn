# importing modules
import os
import joblib

# importing local modules
from config.configs import Config
from src.utils.logger import setup_logger

# at the top of iohelperspy
logger = setup_logger(__name__)
if logger is None:
    # fallback create a basic logger
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("fallback logger used - no file logging available")


def save_model(model, filepath):
    try:
        # check if directory exists create if not
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("created directory: %s", directory)

        # saving the model
        joblib.dump(model, filepath)
        logger.info("model saved successfully to: %s yay!", filepath)
        return True

    except Exception as error:
        logger.error("failed to save model to %s: %s", filepath, str(error))
        return False


def load_model(filepath):
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
    try:
        # preprocessor should have a save method
        preprocessor.save(filepath)
        return True

    except Exception as error:
        logger.error("failed to save preprocessor: %s", str(error))
        return False


def load_preprocessor(filepath=Config.PREPROCESSOR_PATH):
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
    exists = os.path.exists(filepath)
    if exists:
        logger.info("file exists: %s", filepath)
    else:
        logger.warning("file does not exist: %s", filepath)
    return exists


def get_model_paths():
    paths = {
        'custom_model': Config.CUSTOM_MODEL_PATH,
        'sklearn_model': Config.SKLEARN_MODEL_PATH,
        'preprocessor': Config.PREPROCESSOR_PATH
    }
    return paths