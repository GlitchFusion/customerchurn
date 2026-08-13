# create_dirs.py
import os
from config.configs import Config

os.makedirs(os.path.dirname(Config.RAW_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.LOG_PATH), exist_ok=True)
os.makedirs(Config.FIGURES_DIR, exist_ok=True)

print("All directories created! yay!")