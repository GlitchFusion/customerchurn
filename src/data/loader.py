# IMPORTS
import pandas as pd

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_raw_data():
    logger.info("loading raw data from: %s", Config.RAW_DATA_PATH)

    try:
        df = pd.read_csv(Config.RAW_DATA_PATH)
        logger.info("successfully loaded %d rows and %d columns", df.shape[0], df.shape[1])
        return df

    except FileNotFoundError:
        logger.error("file not found at: %s", Config.RAW_DATA_PATH)
        raise

    except pd.errors.EmptyDataError:
        logger.error("the file is empty: %s", Config.RAW_DATA_PATH)
        raise

    except Exception as e:
        logger.error("unexpected error loading data: %s", str(e))
        raise