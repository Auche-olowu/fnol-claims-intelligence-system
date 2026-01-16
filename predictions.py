import streamlit as st
import pandas as pd
import numpy as np

from models import load_model, align_to_schema


def render_predictions(claims_data: pd.DataFrame):
    st.title("🔮 FNOL Prediction")
    st.caption("Estimate the final claim cost at FNOL to support early triage and decision-making.")

    if claims_data is None or claims_data.empty:
        st.warning("No data available. Load claims data first.")
        return

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Enter claim details")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            claim_type_opts = sorted(claims_data["Claim_Type"].dropna().unique()) if "Claim_Type" in claims_data.columns else []
            vehicle_type_opts = sorted(claims_data["Vehicle_Type"].dropna().unique()) if "Vehicle_Type" in claims_data.columns else []
            traffic_opts = sorted(claims_data["Traffic_Condition"].dropna().unique()) if "Traffic_Condition" in claims_data.columns else []
            weather_opts = sorted(claims_data["Weather_Condition"].dropna().unique()) if "Weather_Condition" in claims_data.columns else []

            with c1:
                claim_type = st.selectbox("Claim Type", options=claim_type_opts or ["Collision"])
                traffic = st.selectbox("Traffic Condition", options=traffic_opts or ["Moderate"])
                est_claim = st.number_input("Estimated Claim Amount (£)", min_value=0.0, value=1000.0, step=100.0)

                driver_age = st.number_input("Driver Age", min_value=18, max_value=100, value=35, step=1)
                license_age = st.number_input("License Age (years)", min_value=0, max_value=80, value=10, step=1)

            with c2:
                vehicle_type = st.selectbox("Vehicle Type", options=vehicle_type_opts or ["Sedan"])
                weather = st.selectbox("Weather Condition", options=weather_opts or ["Clear"])
                vehicle_year = st.number_input("Vehicle Year", min_value=1980, max_value=2100, value=2020, step=1)

            submitted = st.form_submit_button("Predict Ultimate Claim Amount", use_container_width=True)

    with right:
        st.subheader("Input summary")
        st.info(
            f"""
**Selected parameters**
- Claim Type: {claim_type}
- Vehicle Type: {vehicle_type}
- Traffic: {traffic}
- Weather: {weather}
- Driver Age: {driver_age}
- License Age: {license_age} years
- Vehicle Year: {vehicle_year}
- Estimated Claim: £{est_claim:,.2f}
"""
        )

    if not submitted:
        return

    st.divider()
    st.subheader("📊 Prediction Results")

    try:
        # Hugging Face production artifacts
        model, feature_columns = load_model()

        # Raw input (must match training feature names)
        input_data = pd.DataFrame(
            {
                "Claim_Type": [str(claim_type)],
                "Estimated_Claim_Amount": [float(est_claim)],
                "Traffic_Condition": [str(traffic)],
                "Weather_Condition": [str(weather)],
                "Vehicle_Type": [str(vehicle_type)],
                "Vehicle_Year": [int(vehicle_year)],
                "Driver_Age": [int(driver_age)],
                "License_Age": [int(license_age)],
            }
        )

        # One-hot encode
        input_encoded = pd.get_dummies(input_data, drop_first=False)

        # Align to training schema
        X_infer = align_to_schema(input_encoded, feature_columns)

        # Predict (log1p -> expm1)
        pred_log = model.predict(X_infer)[0]
        predicted_amount = float(np.expm1(pred_log))

        delta_value = predicted_amount - float(est_claim)
        variance_pct = (delta_value / float(est_claim)) * 100 if float(est_claim) > 0 else 0.0

        colA, colB, colC = st.columns(3)
        colA.metric("Estimated Claim Amount", f"£{float(est_claim):,.2f}")
        colB.metric("Predicted Ultimate Amount", f"£{predicted_amount:,.2f}", delta=f"£{delta_value:,.2f}")
        colC.metric("Variance (%)", f"{variance_pct:.1f}%")

        st.subheader("🧠 FNOL Decision Insight")

        if float(est_claim) <= 0:
            st.info(
                "No FNOL estimate was provided (estimated amount is £0), so variance cannot be interpreted. "
                "Use the predicted ultimate value as the initial benchmark, then refine once an estimate is available."
            )
            return

        abs_var = abs(variance_pct)
        direction = "above" if delta_value > 0 else "below"
        direction_text = f"The model expects the final settlement to be **£{abs(delta_value):,.0f} {direction}** the FNOL estimate."

        if abs_var <= 5:
            st.success(
                f"{direction_text}\n\n"
                "✅ **Low variance**: prediction is close to the FNOL estimate. "
                "This claim may be suitable for straight-through processing."
            )
            severity = "Low"
        elif abs_var <= 15:
            st.warning(
                f"{direction_text}\n\n"
                "⚠️ **Medium variance**: moderate deviation. "
                "Recommend a light-touch review (repair scope, labour/parts, liability)."
            )
            severity = "Medium"
        else:
            st.error(
                f"{direction_text}\n\n"
                "🚨 **High variance**: large deviation. "
                "Prioritise for closer investigation or escalation (complexity/fraud checks/specialist handling)."
            )
            severity = "High"

        st.caption(
            f"Severity band: **{severity}** — variance (%) standardises the gap across claim sizes "
            f"(e.g., a £200 difference is large for a £1,000 claim but small for a £25,000 claim)."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that your Hugging Face repo contains best_model.pkl and feature_columns.pkl.")
