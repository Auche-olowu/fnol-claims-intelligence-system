from pathlib import Path
import pandas as pd
import streamlit as st

from src.feature_engineering import (
    merge_claims_policy,
    add_features
)

from src.preprocessing import (
    fill_missing
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CLAIMS_PATH = DATA_DIR / "claims_table.csv"

POLICY_PATH = DATA_DIR / "policyholder_table.csv"


# ==========================================================
# Load Data
# ==========================================================

@st.cache_data(show_spinner="Loading FNOL datasets...")
def load_claims_and_policy():

    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"Claims dataset not found:\n{CLAIMS_PATH}"
        )

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Policyholder dataset not found:\n{POLICY_PATH}"
        )

    claims_df = pd.read_csv(CLAIMS_PATH)

    policy_df = pd.read_csv(POLICY_PATH)

    merged_df = merge_claims_policy(
        claims_df,
        policy_df,
        how="left"
    )

    merged_df = add_features(merged_df)

    merged_df = fill_missing(
        merged_df,
        report=False
    )

    return claims_df, policy_df, merged_df
