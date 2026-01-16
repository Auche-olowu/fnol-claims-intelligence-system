import streamlit as st
from overview import render_overview
from predictions import render_predictions
from retrain_dashboard import render_retrain_dashboard
from visualizations import render_visualizations
from data_loader import load_claims_and_policy



st.set_page_config(page_title="FNOL Claims Intelligence", layout="wide")


def main():
    st.sidebar.title("🧠 FNOL Claims Intelligence")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Visual Analytics", "Predictions", "Retraining"],
        index=0
    )

    # Retraining can run without loading claims/policy in UI
    if page == "Retraining":
        render_retrain_dashboard()
        return

    # Load data once for all other pages
    try:
        claims_df, policy_df, merged_df = load_claims_and_policy()
        st.sidebar.success("✅ Data loaded automatically")
    except Exception as e:
        st.sidebar.error("❌ Failed to load datasets")
        st.error(str(e))
        st.stop()

    # Route pages
    if page == "Overview":
        render_overview(merged_df)

    elif page == "Visual Analytics":
        render_visualizations(merged_df)

    elif page == "Predictions":
        render_predictions(merged_df)


if __name__ == "__main__":
    main()