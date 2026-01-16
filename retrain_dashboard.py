import streamlit as st
import pandas as pd

from models import retrain_model, load_model


def render_retrain_dashboard():
    st.title("🔁 Model Retraining Dashboard")
    st.caption(
        "Retrain the FNOL claim cost model using new data. "
        "If performance improves, the model is approved and prepared for deployment."
    )

    # ---------------------------------------------------
    # Current Production Model (Hugging Face)
    # ---------------------------------------------------
    with st.expander("📦 Current Production Model (Hugging Face)", expanded=True):
        try:
            model, feature_cols = load_model()
            st.success("✅ Production model and feature schema loaded from Hugging Face")
            st.caption(f"Model expects {len(feature_cols)} input features.")
        except Exception as e:
            st.error("❌ Unable to load production model from Hugging Face")
            st.caption(str(e))
            return

    st.divider()

    # ---------------------------------------------------
    # Upload new training data
    # ---------------------------------------------------
    st.subheader("📤 Upload New Training Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.info("Upload a CSV file to begin retraining.")
        return

    new_data = pd.read_csv(uploaded_file)
    st.dataframe(new_data.head(10), use_container_width=True)

    if "Ultimate_Claim_Amount" not in new_data.columns:
        st.error("❌ Required target column missing: Ultimate_Claim_Amount")
        return

    st.divider()

    # ---------------------------------------------------
    # Retrain & Evaluate
    # ---------------------------------------------------
    st.subheader("🚀 Retrain & Evaluate")

    if st.button("Retrain Model", use_container_width=True):
        with st.spinner("Retraining model and evaluating performance..."):
            result = retrain_model(new_data)

        rmse_old = result["rmse_old"]
        rmse_new = result["rmse_new"]
        promoted = result["promoted"]

        delta = rmse_old - rmse_new
        delta_pct = (delta / rmse_old * 100) if rmse_old else 0

        st.success("Retraining completed.")

        # -----------------------------
        # Metrics
        # -----------------------------
        m1, m2, m3 = st.columns(3)

        m1.metric("Production RMSE (HF)", f"{rmse_old:,.4f}")
        m2.metric(
            "New Model RMSE",
            f"{rmse_new:,.4f}",
            f"{delta:,.4f} ({delta_pct:.2f}%)"
        )
        m3.metric("Approved?", "✅ YES" if promoted else "❌ NO")

        # -----------------------------
        # Promotion messaging
        # -----------------------------
        if promoted:
            st.balloons()
            st.success("🎉 New model approved for production")

            st.info(
                "✅ **Next step: Deployment**\n\n"
                "The new model has outperformed the current Hugging Face production model.\n\n"
                "**If promoted, upload the new model artefacts to Hugging Face to update the deployed production model.**\n\n"
                "This step is intentionally manual to support governance, review, and audit requirements."
            )

            st.caption(
                "Workflow: retrain compares vs Hugging Face → "
                "if better → saves locally → shows *Ready to push* → "
                "manual upload to Hugging Face."
            )
        else:
            st.info(
                "The new model did not outperform the current production model. "
                "No promotion was made."
            )

        with st.expander("📄 View Raw Retraining Output"):
            st.json(result)
