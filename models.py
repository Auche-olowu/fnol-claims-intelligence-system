import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from huggingface_hub import hf_hub_download


# ---------------------------
# Local paths
# ---------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------
# Hugging Face (production model registry)
# ---------------------------
REPO_ID = "Akuoma12/ultimate_claim_cost_model"
MODEL_FILENAME = "best_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"


# ---------------------------
# Core data prep
# ---------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_datetime_safe(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def merge_claims_policy(claims: pd.DataFrame, policy: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    claims = standardize_columns(claims)
    policy = standardize_columns(policy)

    keys = ["Policy_ID", "Customer_ID"]
    for k in keys:
        if k not in claims.columns or k not in policy.columns:
            raise KeyError(f"Missing merge key: {k}")

    return claims.merge(policy, on=keys, how=how)


# ---------------------------
# Feature engineering
# ---------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_cols = ["Accident_Date", "FNOL_Date", "Settlement_Date", "Date_of_Birth", "Full_License_Issue_Date"]
    df = to_datetime_safe(df, date_cols)

    if "Accident_Date" in df.columns and "Date_of_Birth" in df.columns:
        df["Driver_Age"] = (df["Accident_Date"] - df["Date_of_Birth"]).dt.days // 365

    if "Accident_Date" in df.columns and "Full_License_Issue_Date" in df.columns:
        df["License_Age"] = (df["Accident_Date"] - df["Full_License_Issue_Date"]).dt.days // 365

    if "FNOL_Date" in df.columns and "Accident_Date" in df.columns:
        df["FNOL_Delay_Days"] = (df["FNOL_Date"] - df["Accident_Date"]).dt.days

    if "Settlement_Date" in df.columns and "FNOL_Date" in df.columns:
        df["Settlement_Days"] = (df["Settlement_Date"] - df["FNOL_Date"]).dt.days

    # Optional extra features
    if "Accident_Date" in df.columns:
        df["Weekday_Accident"] = df["Accident_Date"].dt.day_name()

        month = df["Accident_Date"].dt.month
        # df["Season_of_Year"] = np.select(
        #     [
        #         month.isin([12, 1, 2]),
        #         month.isin([3, 4, 5]),
        #         month.isin([6, 7, 8]),
        #         month.isin([9, 10, 11]),
        #     ],
        #     ["Winter", "Spring", "Summer", "Autumn"],
        #     default=np.nan,
        # )

    if "Claim_Type" in df.columns:
        high = {"bodily_injury", "theft"}
        medium = {"collision", "fire", "vandalism"}
        low = {"glass", "animal_collision"}

        df["Claim_Type_Risk_Category"] = np.select(
            [df["Claim_Type"].isin(high), df["Claim_Type"].isin(medium), df["Claim_Type"].isin(low)],
            ["High", "Medium", "Low"],
            default="Other",
        )

    return df


# ---------------------------
# Missing values
# ---------------------------
def fill_missing(df: pd.DataFrame, report: bool = False) -> pd.DataFrame:
    df = df.copy()

    num_cols = [
        "Estimated_Claim_Amount",
        "Ultimate_Claim_Amount",
        "FNOL_Delay_Days",
        "Settlement_Days",
        "Driver_Age",
        "License_Age",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            med = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(med) else med)

    cat_cols = [
        "Traffic_Condition",
        "Weather_Condition",
        "Vehicle_Type",
        "Claim_Type",
        "Season_of_Year",
        "Claim_Type_Risk_Category",
        "Weekday_Accident",
    ]
    for col in cat_cols:
        if col in df.columns:
            mode_s = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_s.iloc[0] if len(mode_s) else "Unknown")

    if report:
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            print("Remaining missing values:\n", missing.sort_values(ascending=False))

    return df


# ---------------------------
# Outlier clipping (simple)
# ---------------------------
def winsorize_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        return df

    s = pd.to_numeric(df[column], errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[column] = s.clip(lower, upper)
    return df


# ---------------------------
# Encoding + schema alignment
# ---------------------------
def one_hot_encode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=False, dtype=int)


def save_feature_columns(feature_columns: list) -> None:
    _ensure_models_dir()
    joblib.dump(list(feature_columns), FEATURES_PATH)


def load_feature_columns() -> list:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature schema not found at {FEATURES_PATH}")
    return joblib.load(FEATURES_PATH)


def align_to_schema(X: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Ensures input columns match the model's expected training schema:
    - Adds missing columns as 0
    - Drops extra columns
    - Orders columns correctly
    """
    X = X.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)
    return X[feature_columns]


# ---------------------------
# Save/Load model
# ---------------------------
def save_model(model, versioned: bool = False) -> str:
    """
    Keeps your local saving behaviour (for retraining experiments).
    NOTE: This does NOT push to Hugging Face automatically.
    """
    _ensure_models_dir()

    if versioned:
        v = 1
        while os.path.exists(os.path.join(MODELS_DIR, f"best_model_v{v}.pkl")):
            v += 1
        path = os.path.join(MODELS_DIR, f"best_model_v{v}.pkl")
    else:
        path = MODEL_PATH

    joblib.dump(model, path)
    return path


def load_model():
    """
    ✅ Production load comes from Hugging Face Hub.
    Returns: (model, feature_columns)
    """
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    features_path = hf_hub_download(repo_id=REPO_ID, filename=FEATURES_FILENAME)

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    return model, feature_columns


def load_model_local():
    """
    Optional local fallback load (useful during development / retrain comparisons).
    Returns: (model, feature_columns)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Local model not found at {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Local feature schema not found at {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


# ---------------------------
# Training helpers
# ---------------------------
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

TARGET = "Ultimate_Claim_Amount"
CATEGORICAL_FEATURES = ["Traffic_Condition", "Weather_Condition", "Vehicle_Type", "Claim_Type"]


def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    df = fill_missing(df)

    for col in ["Estimated_Claim_Amount", "Ultimate_Claim_Amount", "Driver_Age", "License_Age"]:
        if col in df.columns:
            df = winsorize_iqr(df, col)

    keep = [c for c in FEATURES_BASE + [TARGET] if c in df.columns]
    return df[keep].copy()


def train_production_model(train_df: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Trains a production-style model locally and saves:
    - models/best_model.pkl
    - models/feature_columns.pkl
    """
    df = prepare_for_model(train_df)

    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' missing from training data.")

    df[TARGET] = np.log1p(df[TARGET])
    df = one_hot_encode(df, CATEGORICAL_FEATURES)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    save_feature_columns(X_train.columns.tolist())

    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    save_model(model, versioned=False)
    return model, {"rmse": rmse}


def retrain_model(new_data: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Retrains locally and compares to current local production model.
    Promotes if RMSE improves.
    """
    prod_model, expected_cols = load_model_local()

    df = prepare_for_model(new_data)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' missing from retraining data.")

    df[TARGET] = np.log1p(df[TARGET])
    df = one_hot_encode(df, CATEGORICAL_FEATURES)

    df = df.reindex(columns=expected_cols + [TARGET], fill_value=0)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # evaluate old
    y_pred_old_log = prod_model.predict(X_test)
    rmse_old = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_old_log), squared=False)

    # train new with same params
    params = prod_model.get_params() if hasattr(prod_model, "get_params") else {}
    new_model = RandomForestRegressor(**params)
    new_model.fit(X_train, y_train)

    y_pred_new_log = new_model.predict(X_test)
    rmse_new = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_new_log), squared=False)

    promoted = False
    if rmse_new < rmse_old:
        save_model(new_model, versioned=False)
        save_feature_columns(expected_cols)
        promoted = True

    return {"rmse_old": rmse_old, "rmse_new": rmse_new, "promoted": promoted}
