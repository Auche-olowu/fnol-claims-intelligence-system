"""
==========================================================
Feature Engineering Module

FNOL Claims Intelligence Platform
==========================================================
"""

import numpy as np
import pandas as pd


# ==========================================================
# Standardise Column Names
# ==========================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove whitespace from column names.
    """

    df = df.copy()

    df.columns = [str(col).strip() for col in df.columns]

    return df


# ==========================================================
# Safe Datetime Conversion
# ==========================================================

def to_datetime_safe(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:

    """
    Convert date columns safely.
    """

    df = df.copy()

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


# ==========================================================
# Merge Claims + Policyholder Tables
# ==========================================================

def merge_claims_policy(
    claims_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    how: str = "left"
) -> pd.DataFrame:

    claims_df = standardize_columns(claims_df)

    policy_df = standardize_columns(policy_df)

    merge_keys = [

        "Policy_ID",

        "Customer_ID"

    ]

    for key in merge_keys:

        if key not in claims_df.columns:

            raise KeyError(f"{key} missing from Claims dataset.")

        if key not in policy_df.columns:

            raise KeyError(f"{key} missing from Policyholder dataset.")

    return claims_df.merge(

        policy_df,

        on=merge_keys,

        how=how

    )


# ==========================================================
# Feature Engineering
# ==========================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    Create derived machine learning features.
    """

    df = df.copy()

    date_columns = [

        "Accident_Date",

        "FNOL_Date",

        "Settlement_Date",

        "Date_of_Birth",

        "Full_License_Issue_Date"

    ]

    df = to_datetime_safe(df, date_columns)

    # Driver Age

    if {

        "Accident_Date",

        "Date_of_Birth"

    }.issubset(df.columns):

        df["Driver_Age"] = (

            df["Accident_Date"]

            - df["Date_of_Birth"]

        ).dt.days // 365

    # Licence Age

    if {

        "Accident_Date",

        "Full_License_Issue_Date"

    }.issubset(df.columns):

        df["License_Age"] = (

            df["Accident_Date"]

            - df["Full_License_Issue_Date"]

        ).dt.days // 365

    # FNOL Delay

    if {

        "FNOL_Date",

        "Accident_Date"

    }.issubset(df.columns):

        df["FNOL_Delay_Days"] = (

            df["FNOL_Date"]

            - df["Accident_Date"]

        ).dt.days

    # Settlement Duration

    if {

        "Settlement_Date",

        "FNOL_Date"

    }.issubset(df.columns):

        df["Settlement_Days"] = (

            df["Settlement_Date"]

            - df["FNOL_Date"]

        ).dt.days

    # Vehicle Age

    if {

        "Vehicle_Year",

        "Accident_Date"

    }.issubset(df.columns):

        df["Vehicle_Age"] = (

            df["Accident_Date"].dt.year

            - df["Vehicle_Year"]

        )

    # Weekday

    if "Accident_Date" in df.columns:

        df["Weekday_Accident"] = (

            df["Accident_Date"]

            .dt.day_name()

        )

    # Season

    if "Accident_Date" in df.columns:

        month = df["Accident_Date"].dt.month

        df["Season"] = np.select(

            [

                month.isin([12,1,2]),

                month.isin([3,4,5]),

                month.isin([6,7,8]),

                month.isin([9,10,11])

            ],

            [

                "Winter",

                "Spring",

                "Summer",

                "Autumn"

            ],

            default="Unknown"

        )

    # Claim Risk

    if "Claim_Type" in df.columns:

        high = {

            "bodily_injury",

            "theft"

        }

        medium = {

            "collision",

            "fire",

            "vandalism"

        }

        low = {

            "glass",

            "animal_collision"

        }

        df["Claim_Type_Risk_Category"] = np.select(

            [

                df["Claim_Type"].isin(high),

                df["Claim_Type"].isin(medium),

                df["Claim_Type"].isin(low)

            ],

            [

                "High",

                "Medium",

                "Low"

            ],

            default="Other"

        )

    return df