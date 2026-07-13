"""
==========================================================
Model Loader Module

FNOL Claims Intelligence Platform
==========================================================

Responsible for:

• Saving trained models
• Loading production models
• Loading local models
• Downloading models from Hugging Face

==========================================================
"""

import joblib

from huggingface_hub import hf_hub_download

from src.config import (
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    MODELS_DIR,
    HF_REPO_ID,
    HF_MODEL_FILE,
    HF_FEATURE_FILE
)


# ==========================================================
# Save Model
# ==========================================================

def save_model(
    model,
    versioned=False
):

    MODELS_DIR.mkdir(
        exist_ok=True
    )

    if versioned:

        version = 1

        while (MODELS_DIR / f"best_model_v{version}.pkl").exists():

            version += 1

        save_path = MODELS_DIR / f"best_model_v{version}.pkl"

    else:

        save_path = MODEL_PATH

    joblib.dump(
        model,
        save_path
    )

    return save_path


# ==========================================================
# Save Feature Columns
# ==========================================================

def save_feature_columns(
    columns
):

    MODELS_DIR.mkdir(
        exist_ok=True
    )

    joblib.dump(
        list(columns),
        FEATURE_COLUMNS_PATH
    )


# ==========================================================
# Load Feature Columns
# ==========================================================

def load_feature_columns():

    return joblib.load(
        FEATURE_COLUMNS_PATH
    )


# ==========================================================
# Load Local Model
# ==========================================================

def load_model_local():

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    return model, feature_columns


# ==========================================================
# Load Production Model
# ==========================================================

def load_model():

    model_file = hf_hub_download(

        repo_id=HF_REPO_ID,

        filename=HF_MODEL_FILE

    )

    feature_file = hf_hub_download(

        repo_id=HF_REPO_ID,

        filename=HF_FEATURE_FILE

    )

    model = joblib.load(
        model_file
    )

    feature_columns = joblib.load(
        feature_file
    )

    return model, feature_columns