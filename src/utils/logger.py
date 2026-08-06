# Creating logs to keep track of what's happening BTS

import logging
import sys
from config.configs import Config

def log_setup(name = __name__, log_file = Config.LOGS_DIR):
    level=logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger


    # formatter
    formatter = logging.Formatter(
        "%(asctime)s \n %(levelname)s \n %(message)s \n",
        datefmt = "%d-%b-%Y || %H:%M:%S"
    )

    # Writing in the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Writing in the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)