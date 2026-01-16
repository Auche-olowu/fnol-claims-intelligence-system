import os
import pandas as pd
import streamlit as st

from models import merge_claims_policy, add_features, fill_missing

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "FNOL_DATASET")

CLAIMS_PATH = os.path.join(DATA_DIR, "claims_table.csv")
POLICY_PATH = os.path.join(DATA_DIR, "policyholder_table.csv")


@st.cache_data(show_spinner=True)
def load_claims_and_policy():
    if not os.path.exists(CLAIMS_PATH):
        raise FileNotFoundError(f"Claims dataset not found: {CLAIMS_PATH}")

    if not os.path.exists(POLICY_PATH):
        raise FileNotFoundError(f"Policy dataset not found: {POLICY_PATH}")

    claims_df = pd.read_csv(CLAIMS_PATH)
    policy_df = pd.read_csv(POLICY_PATH)

    merged_df = merge_claims_policy(claims_df, policy_df, how="left")
    merged_df = add_features(merged_df)
    merged_df = fill_missing(merged_df, report=False)

    return claims_df, policy_df, merged_df
