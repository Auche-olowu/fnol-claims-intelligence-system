"""
==========================================================
Preprocessing Module

FNOL Claims Intelligence Platform
==========================================================
"""

import joblib
import pandas as pd

from src.config import (
    FEATURE_COLUMNS_PATH,
    TARGET,
    FEATURES_BASE,
    CATEGORICAL_FEATURES
)


# ==========================================================
# Missing Values
# ==========================================================

def fill_missing(
    df: pd.DataFrame,
    report: bool = False
) -> pd.DataFrame:

    df = df.copy()

    numeric_columns = [

        "Estimated_Claim_Amount",

        "Ultimate_Claim_Amount",

        "FNOL_Delay_Days",

        "Settlement_Days",

        "Driver_Age",

        "License_Age",

        "Vehicle_Age"

    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

            median = df[col].median()

            df[col] = df[col].fillna(

                0 if pd.isna(median) else median

            )

    categorical_columns = [

        "Traffic_Condition",

        "Weather_Condition",

        "Vehicle_Type",

        "Claim_Type",

        "Season",

        "Weekday_Accident",

        "Claim_Type_Risk_Category"

    ]

    for col in categorical_columns:

        if col in df.columns:

            mode = df[col].mode(dropna=True)

            df[col] = df[col].fillna(

                mode.iloc[0] if len(mode) else "Unknown"

            )

    if report:

        missing = df.isna().sum()

        missing = missing[missing > 0]

        if len(missing):

            print(missing)

    return df


# ==========================================================
# Winsorisation
# ==========================================================

def winsorize_iqr(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:

    df = df.copy()

    if column not in df.columns:

        return df

    q1 = df[column].quantile(0.25)

    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    df[column] = df[column].clip(

        lower,

        upper

    )

    return df


# ==========================================================
# Encoding
# ==========================================================

def one_hot_encode(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    available = [

        col

        for col in CATEGORICAL_FEATURES

        if col in df.columns

    ]

    return pd.get_dummies(

        df,

        columns=available,

        drop_first=False,

        dtype=int

    )


# ==========================================================
# Feature Schema
# ==========================================================

def save_feature_columns(

    columns

):

    joblib.dump(

        list(columns),

        FEATURE_COLUMNS_PATH

    )


def load_feature_columns():

    return joblib.load(

        FEATURE_COLUMNS_PATH

    )


def align_to_schema(

    X,

    feature_columns

):

    X = X.copy()

    for col in feature_columns:

        if col not in X.columns:

            X[col] = 0

    X = X[feature_columns]

    return X


# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_for_model(

    df: pd.DataFrame

) -> pd.DataFrame:

    df = fill_missing(df)

    for col in [

        "Estimated_Claim_Amount",

        "Ultimate_Claim_Amount",

        "Driver_Age",

        "License_Age",

        "Vehicle_Age"

    ]:

        if col in df.columns:

            df = winsorize_iqr(

                df,

                col

            )

    keep = [

        c

        for c in FEATURES_BASE + [TARGET]

        if c in df.columns

    ]

    return df[keep].copy()