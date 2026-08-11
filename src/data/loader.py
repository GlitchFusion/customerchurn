import pandas as pd
from config.configs import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data():
    logger.info(f"Loading data from {Config.RAW_DATA_DIR}")

    try:
        df = pd.read_csv(Config.RAW_DATA_DIR)
        logger.info(f"DATA LOADED SUCCESSFULLY \n DIMENSION: {df.shape} \n ROWS: {df.shape[0]} \n COLUMNS: {df.shape[1]}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {Config.RAW_DATA_DIR}. \n RECHECK THE FILE PATH. \n ERROR: {e}")
        raise e

    except pd.errors.EmptyDataError as e:
        logger.error(f"Sorry the data can't be located at: {Config.RAW_DATA_DIR}. \n RECHECK THE FILE PATH. \n ERROR: {e}")

    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise e