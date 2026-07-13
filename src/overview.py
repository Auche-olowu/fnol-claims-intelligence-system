import streamlit as st
import pandas as pd
import plotly.express as px

from src.feature_engineering import add_features
from src.preprocessing import fill_missing


def render_overview(df):

    # ==========================================================
    # Dashboard Header
    # ==========================================================

    st.title("🚗 Executive Claims Dashboard")

    st.markdown("""
    Welcome to the **FNOL Claims Intelligence Platform**.

    This dashboard provides an executive overview of claim volumes,
    claim severity, reporting efficiency and operational performance.

    Use the filters on the left to explore the portfolio.
    """)

    if df is None or df.empty:
        st.warning("No claims available.")
        return

    # ==========================================================
    # Feature Engineering
    # ==========================================================

    df = add_features(df)

    df = fill_missing(df, report=False)

    # ==========================================================
    # Sidebar Filters
    # ==========================================================

    st.sidebar.header("🔎 Dashboard Filters")

    filtered = df.copy()

    if "Claim_Type" in df.columns:

        claim_types = ["All"] + sorted(
            df["Claim_Type"].dropna().unique()
        )

        selected_claim = st.sidebar.selectbox(
            "Claim Type",
            claim_types
        )

        if selected_claim != "All":

            filtered = filtered[
                filtered["Claim_Type"] == selected_claim
            ]

    if "Vehicle_Type" in df.columns:

        vehicle_types = ["All"] + sorted(
            df["Vehicle_Type"].dropna().unique()
        )

        selected_vehicle = st.sidebar.selectbox(
            "Vehicle Type",
            vehicle_types
        )

        if selected_vehicle != "All":

            filtered = filtered[
                filtered["Vehicle_Type"] == selected_vehicle
            ]

    if filtered.empty:

        st.warning(
            "No records found for the selected filters."
        )

        return

    # ==========================================================
    # Executive KPI Cards
    # ==========================================================

    st.markdown("---")

    st.subheader("📊 Executive Summary")

    kpi1, kpi2, kpi3 = st.columns(3)

    kpi4, kpi5, kpi6 = st.columns(3)

    # Total Claims

    kpi1.metric(

        "Total Claims",

        f"{len(filtered):,}"

    )

    # Average Ultimate

    if "Ultimate_Claim_Amount" in filtered.columns:

        avg_ultimate = filtered[
            "Ultimate_Claim_Amount"
        ].mean()

        kpi2.metric(

            "Average Ultimate",

            f"£{avg_ultimate:,.0f}"

        )

    else:

        avg_ultimate = 0

        kpi2.metric(

            "Average Ultimate",

            "N/A"

        )

    # Average Estimated

    if "Estimated_Claim_Amount" in filtered.columns:

        avg_estimated = filtered[
            "Estimated_Claim_Amount"
        ].mean()

        kpi3.metric(

            "Average Estimated",

            f"£{avg_estimated:,.0f}"

        )

    else:

        avg_estimated = 0

        kpi3.metric(

            "Average Estimated",

            "N/A"

        )

    # FNOL Delay

    if "FNOL_Delay_Days" in filtered.columns:

        avg_delay = filtered[
            "FNOL_Delay_Days"
        ].mean()

        kpi4.metric(

            "Average FNOL Delay",

            f"{avg_delay:.1f} Days"

        )

    else:

        avg_delay = 0

        kpi4.metric(

            "Average FNOL Delay",

            "N/A"

        )

    # Driver Age

    if "Driver_Age" in filtered.columns:

        avg_driver = filtered[
            "Driver_Age"
        ].mean()

        kpi5.metric(

            "Average Driver Age",

            f"{avg_driver:.0f}"

        )

    else:

        kpi5.metric(

            "Average Driver Age",

            "N/A"

        )

    # Vehicle Age

    if "Vehicle_Age" in filtered.columns:

        avg_vehicle = filtered[
            "Vehicle_Age"
        ].mean()

        kpi6.metric(

            "Average Vehicle Age",

            f"{avg_vehicle:.1f} Years"

        )

    else:

        kpi6.metric(

            "Average Vehicle Age",

            "N/A"

        )

    # ==========================================================
    # Business Health
    # ==========================================================

    st.markdown("---")

    st.subheader("📈 Portfolio Health")

    left, middle, right = st.columns(3)

    inflation = avg_ultimate - avg_estimated

    left.metric(

        "Average Claim Inflation",

        f"£{inflation:,.0f}"

    )

    health = max(

        0,

        min(

            100,

            round(

                100

                - avg_delay * 4

                - inflation / 2000,

                1

            )

        )

    )

    middle.metric(

        "Business Health",

        f"{health}/100"

    )

    right.metric(

        "Production Model",

        "Random Forest"

    )

    st.progress(

        health / 100

    )

    st.markdown("---")

    # -------------------------
    # Claim Type Analysis
    # -------------------------
        # ==========================================================
    # Monthly Claims Trend
    # ==========================================================

    st.subheader("📈 Claims Trend")

    if "Accident_Date" in filtered.columns:

        filtered["Accident_Date"] = pd.to_datetime(
            filtered["Accident_Date"],
            errors="coerce"
        )

        monthly = (
            filtered
            .dropna(subset=["Accident_Date"])
            .assign(
                Month=lambda x:
                x["Accident_Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month")
            .size()
            .reset_index(name="Claims")
        )

        fig = px.line(
            monthly,
            x="Month",
            y="Claims",
            markers=True,
            title="Monthly Claim Volume"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================================
    # Claim Type Summary
    # ==========================================================

    st.subheader("🧾 Claim Type Summary")

    if {
        "Claim_Type",
        "Estimated_Claim_Amount",
        "Ultimate_Claim_Amount"
    }.issubset(filtered.columns):

        claim_summary = (

            filtered

            .groupby("Claim_Type")

            .agg(

                Claims=("Claim_Type","count"),

                Average_Estimated=(
                    "Estimated_Claim_Amount",
                    "mean"
                ),

                Average_Ultimate=(
                    "Ultimate_Claim_Amount",
                    "mean"
                ),

                Total_Ultimate=(
                    "Ultimate_Claim_Amount",
                    "sum"
                )

            )

            .round(2)

            .sort_values(
                "Total_Ultimate",
                ascending=False
            )

            .reset_index()

        )

        left,right = st.columns(2)

        with left:

            fig = px.bar(

                claim_summary,

                x="Claim_Type",

                y="Total_Ultimate",

                title="Total Ultimate Cost by Claim Type"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.bar(

                claim_summary,

                x="Claim_Type",

                y="Average_Ultimate",

                title="Average Ultimate Cost"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with st.expander("View Claim Type Statistics"):

            st.dataframe(
                claim_summary,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # ==========================================================
    # Risk Factor Analysis
    # ==========================================================

    st.subheader("⚠️ Risk Factors")

    col1,col2 = st.columns(2)

    if "Weather_Condition" in filtered.columns:

        weather = (

            filtered

            .groupby("Weather_Condition")

            ["Ultimate_Claim_Amount"]

            .mean()

            .reset_index()

        )

        fig = px.bar(

            weather,

            x="Weather_Condition",

            y="Ultimate_Claim_Amount",

            title="Average Ultimate Cost by Weather"

        )

        col1.plotly_chart(
            fig,
            use_container_width=True
        )

    if "Traffic_Condition" in filtered.columns:

        traffic = (

            filtered

            .groupby("Traffic_Condition")

            ["Ultimate_Claim_Amount"]

            .mean()

            .reset_index()

        )

        fig = px.bar(

            traffic,

            x="Traffic_Condition",

            y="Ultimate_Claim_Amount",

            title="Average Ultimate Cost by Traffic"

        )

        col2.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================================
    # Vehicle Analysis
    # ==========================================================

    st.subheader("🚗 Vehicle Insights")

    col1,col2 = st.columns(2)

    if "Vehicle_Type" in filtered.columns:

        vehicle = (

            filtered

            .groupby("Vehicle_Type")

            ["Ultimate_Claim_Amount"]

            .mean()

            .reset_index()

        )

        fig = px.bar(

            vehicle,

            x="Vehicle_Type",

            y="Ultimate_Claim_Amount",

            title="Average Ultimate Cost by Vehicle"

        )

        col1.plotly_chart(
            fig,
            use_container_width=True
        )

    if "Vehicle_Age" in filtered.columns:

        fig = px.histogram(

            filtered,

            x="Vehicle_Age",

            nbins=20,

            title="Vehicle Age Distribution"

        )

        col2.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================================
    # Executive Summary
    # ==========================================================

    st.subheader("📋 Executive Insights")

    highest_claim = filtered["Ultimate_Claim_Amount"].max()

    avg_claim = filtered["Ultimate_Claim_Amount"].mean()

    avg_delay = filtered["FNOL_Delay_Days"].mean()

    st.info(f"""

        ### Portfolio Summary

        • Total Claims Processed: **{len(filtered):,}**

        • Average Ultimate Claim: **£{avg_claim:,.0f}**

        • Highest Claim Recorded: **£{highest_claim:,.0f}**

        • Average FNOL Reporting Delay: **{avg_delay:.1f} days**

        ### Key Business Observations

        - Claims reporting remains efficient when delays are low.

        - Large differences between Estimated and Ultimate Claim values suggest opportunities
        to improve FNOL prediction accuracy.

        - Vehicle type, weather conditions and traffic conditions all contribute to
        differences in claim severity.

        - This dashboard enables claims teams to identify high-risk claim categories
        and improve early intervention strategies.

        """)
    
     # ==========================================================
    # Executive Recommendations
    # ==========================================================

    st.markdown("---")

    st.subheader("🎯 Executive Recommendations")

    recommendations = []

    if inflation > 3000:
        recommendations.append(
            "Large differences exist between estimated and ultimate claim costs. Consider improving FNOL severity prediction using additional claim features."
        )

    if avg_delay > 5:
        recommendations.append(
            "Average FNOL reporting delay exceeds five days. Earlier claim reporting could reduce settlement costs."
        )

    if avg_claim > 20000:
        recommendations.append(
            "Average claim severity is high. Introduce early fraud screening and specialist handling for high-value claims."
        )

    if "Vehicle_Age" in filtered.columns:

        if filtered["Vehicle_Age"].mean() > 10:

            recommendations.append(
                "Older vehicles dominate the portfolio. Consider introducing age-based risk pricing."
            )

    if "Driver_Age" in filtered.columns:

        if filtered["Driver_Age"].mean() < 25:

            recommendations.append(
                "A relatively young driver population may increase claim frequency. Consider targeted driver education programmes."
            )

    if len(recommendations) == 0:

        recommendations.append(
            "Current portfolio performance appears stable. Continue monitoring claim severity and reporting behaviour."
        )

    for rec in recommendations:

        st.success(rec)
        
        
    # ==========================================================
    # Operational Alerts
    # ==========================================================

    st.markdown("---")

    st.subheader("🚨 Operational Alerts")

    alert1, alert2, alert3 = st.columns(3)

    if avg_delay > 5:

        alert1.error("High FNOL Reporting Delay")

    else:

        alert1.success("FNOL Reporting On Target")

    if inflation > 3000:

        alert2.warning("High Claim Inflation")

    else:

        alert2.success("Claim Inflation Stable")

    if avg_claim > 25000:

        alert3.error("High Severity Portfolio")

    else:

        alert3.success("Claim Severity Acceptable")
        
    # ==========================================================
    # Portfolio Scorecard
    # ==========================================================

    st.markdown("---")

    st.subheader("📋 Portfolio Scorecard")

    scorecard = pd.DataFrame({

        "Metric":[

            "Total Claims",

            "Average Ultimate Claim",

            "Average Estimated Claim",

            "Average FNOL Delay",

            "Claim Inflation",

            "Business Health"

        ],

        "Value":[

            f"{len(filtered):,}",

            f"£{avg_claim:,.0f}",

            f"£{avg_estimated:,.0f}",

            f"{avg_delay:.1f} Days",

            f"£{inflation:,.0f}",

            f"{health}/100"

        ]

    })

    st.dataframe(

        scorecard,

        use_container_width=True,

        hide_index=True

    )