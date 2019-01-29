from pathlib import Path
import os

"""
The working directory and subdirectories
"""

ROOT_DIR = Path(__file__).resolve().parent

DATA_SUBDIR = os.path.join(ROOT_DIR, '../data')
GLOVE_SUBDIR = os.path.join(DATA_SUBDIR, 'gloves')
MODEL_SUBDIR = os.path.join(ROOT_DIR, '../model')
