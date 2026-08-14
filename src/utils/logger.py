# IMPORTS
import logging
import sys
import os

# LOCAL IMPORTS
from config.configs import Config


class PathFilter(logging.Filter):
    def filter(self, record):
        # get the root directory path
        root_dir = Config.ROOT_DIR
        
        # if the message contains the full path, replace it
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # replace full root path with just the relative part
            record.msg = record.msg.replace(root_dir, ".")
            
            # also handle path in extra arguments if they exist
            if hasattr(record, 'args') and record.args:
                # args could be a tuple, list, or dict
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        # replace path in each string argument
                        arg = arg.replace(root_dir, ".")
                    new_args.append(arg)
                record.args = tuple(new_args)
        
        return True


def setup_logger(name=__name__, log_file=Config.LOG_PATH):
    # create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers
    if logger.handlers:
        return logger

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # file handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # add path filter to file handler
        file_handler.addFilter(PathFilter())
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"warning: could not create file handler: {e}")

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add path filter to console handler
    console_handler.addFilter(PathFilter())
    logger.addHandler(console_handler)

    return logger