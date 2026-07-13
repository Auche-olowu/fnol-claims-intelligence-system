import streamlit as st
from src.overview import render_overview
from src.prediction import render_predictions
from src.retrain_dashboard import render_retrain_dashboard
from src.visualizations import render_visualizations
from src.data_loader import load_claims_and_policy

from src.config import (
    APP_TITLE,
    PAGE_ICON,
    LAYOUT
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)


def main():
    st.sidebar.title("🧠 FNOL Claims Intelligence")

    page = st.sidebar.radio(
        "Navigation",
        ["Executive Dashboard", "Claims Analytics", "Claim Prediction", "Model Retraining"],
        index=0
    )

    # Retraining can run without loading claims/policy in UI
    if page == "Model Retraining":
        render_retrain_dashboard()
        return

    # Load data once for all other pages
    try:
        with st.spinner("Loading FNOL datasets..."):
            claims_df, policy_df, merged_df = load_claims_and_policy()
            st.sidebar.success("✅ Data loaded automatically")
    except Exception as e:
        st.sidebar.error("❌ Failed to load datasets")
        st.error(
            f"""
            Unable to load the FNOL datasets.
            Reason:
            {e}
            """
            )
        st.stop()

    # Route pages
    if page == "Executive Dashboard":
        render_overview(merged_df)

    elif page == "Claims Analytics":
        render_visualizations(merged_df)

    elif page == "Claim Prediction":
        render_predictions(merged_df)


if __name__ == "__main__":
    main()