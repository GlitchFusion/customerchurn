"""
eda.py - Contains functions for exploatory data analysis on the provided dataset.
Generate text summaries, statistical insights, and visualizations

Purpose of this module: To understand the raw data before preprocessing 
"""

# IMPROTS
import pandas as pd
import numpy as np

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger
