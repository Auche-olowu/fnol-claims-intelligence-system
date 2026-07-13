"""
==========================================================
FNOL Claims Intelligence Platform
Configuration File
==========================================================

Stores all project constants, paths and model settings.

Author: Amanda Olowu
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"

MLFLOW_DIR = PROJECT_ROOT / "mlruns"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

ASSETS_DIR = PROJECT_ROOT / "assets"

SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

# ==========================================================
# Dataset Files
# ==========================================================

CLAIMS_DATA = DATA_DIR / "claims_table.csv"

POLICY_DATA = DATA_DIR / "policyholder_table.csv"

# ==========================================================
# Model Files
# ==========================================================

MODEL_PATH = MODELS_DIR / "best_model.pkl"

FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# ==========================================================
# Hugging Face Repository
# ==========================================================

HF_REPO_ID = "Akuoma12/ultimate_claim_cost_model"

HF_MODEL_FILE = "best_model.pkl"

HF_FEATURE_FILE = "feature_columns.pkl"

# ==========================================================
# Machine Learning Parameters
# ==========================================================

TARGET = "Ultimate_Claim_Amount"

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_ESTIMATORS = 200

N_JOBS = -1

# ==========================================================
# Feature Lists
# ==========================================================

FEATURES_BASE = [

    "Claim_Type",

    "Estimated_Claim_Amount",

    "Traffic_Condition",

    "Weather_Condition",

    "Vehicle_Type",

    "Vehicle_Year",

    "Driver_Age",

    "License_Age",

]

CATEGORICAL_FEATURES = [

    "Traffic_Condition",

    "Weather_Condition",

    "Vehicle_Type",

    "Claim_Type",

]

# ==========================================================
# Dashboard
# ==========================================================

APP_TITLE = "FNOL Claims Intelligence Platform"

PAGE_ICON = "🚗"

LAYOUT = "wide"