import streamlit as st
import pandas as pd
import plotly.express as px

from models import add_features, fill_missing


def render_overview(df: pd.DataFrame):
    st.title("🚗 FNOL Claims Overview")
    st.caption("Key FNOL metrics and factors to support early claim decisions.")

    if df is None or df.empty:
        st.warning("No data available.")
        return

    # Your merged_df already exists, so just add features + fill missing
    df = add_features(df)
    df = fill_missing(df, report=False)

    # -------------------------
    # Sidebar Filters
    # -------------------------
    with st.sidebar:
        st.header("🔎 Filters")

        # Claim type dropdown (All + single select)
        if "Claim_Type" in df.columns:
            claim_type_options = ["All"] + sorted(df["Claim_Type"].dropna().unique().tolist())
            selected_claim_type = st.selectbox("Claim Type", claim_type_options, index=0)
        else:
            selected_claim_type = "All"

        # Vehicle type dropdown (All + single select)
        if "Vehicle_Type" in df.columns:
            vehicle_type_options = ["All"] + sorted(df["Vehicle_Type"].dropna().unique().tolist())
            selected_vehicle_type = st.selectbox("Vehicle Type", vehicle_type_options, index=0)
        else:
            selected_vehicle_type = "All"

    filtered = df.copy()
    if selected_claim_type != "All" and "Claim_Type" in filtered.columns:
        filtered = filtered[filtered["Claim_Type"] == selected_claim_type]

    if selected_vehicle_type != "All" and "Vehicle_Type" in filtered.columns:
        filtered = filtered[filtered["Vehicle_Type"] == selected_vehicle_type]

    if filtered.empty:
        st.warning("No data matches your filters. Please broaden your selection.")
        return

    # -------------------------
    # KPIs (safe in case column missing)
    # -------------------------
    st.subheader("📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Claims", f"{len(filtered):,}")

    if "Ultimate_Claim_Amount" in filtered.columns:
        c2.metric("Average Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].mean():,.0f}")
        c3.metric("Minimum Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].min():,.0f}")
        c4.metric("Maximum Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].max():,.0f}")
    else:
        c2.metric("Average Ultimate (£)", "N/A")
        c3.metric("Minimum Ultimate (£)", "N/A")
        c4.metric("Maximum Ultimate (£)", "N/A")

    st.divider()

    # -------------------------
    # Claim Type Analysis
    # -------------------------
    st.subheader("🧾 Claim Type Analysis")

    required_cols = {"Claim_Type", "Estimated_Claim_Amount", "Ultimate_Claim_Amount"}
    if required_cols.issubset(set(filtered.columns)):
        claim_type_summary = (
            filtered.groupby("Claim_Type")
            .agg(
                Est_Total=("Estimated_Claim_Amount", "sum"),
                Est_Avg=("Estimated_Claim_Amount", "mean"),
                Claim_Count=("Claim_Type", "size"),
                Ult_Total=("Ultimate_Claim_Amount", "sum"),
                Ult_Avg=("Ultimate_Claim_Amount", "mean"),
            )
            .round(2)
            .reset_index()
            .sort_values("Ult_Total", ascending=False)
        )

        highest = claim_type_summary.iloc[0]
        lowest = claim_type_summary.iloc[-1]

        left, right = st.columns(2)
        with left:
            st.success("🔺 Highest total ultimate cost claim type")
            st.write(f"**{highest['Claim_Type']}**")
            st.write(f"Ultimate Total: **£{highest['Ult_Total']:,.0f}**")
            st.write(f"Ultimate Avg: **£{highest['Ult_Avg']:,.0f}**")
            st.write(f"Est Avg (FNOL): **£{highest['Est_Avg']:,.0f}**")
            st.write(f"Claims: **{int(highest['Claim_Count']):,}**")

        with right:
            st.info("🔻 Lowest total ultimate cost claim type")
            st.write(f"**{lowest['Claim_Type']}**")
            st.write(f"Ultimate Total: **£{lowest['Ult_Total']:,.0f}**")
            st.write(f"Ultimate Avg: **£{lowest['Ult_Avg']:,.0f}**")
            st.write(f"Est Avg (FNOL): **£{lowest['Est_Avg']:,.0f}**")
            st.write(f"Claims: **{int(lowest['Claim_Count']):,}**")

        # Validation
        if int(claim_type_summary["Claim_Count"].sum()) == len(filtered):
            st.success("Validation check passed: claim counts match filtered rows.")
        else:
            st.warning("Validation warning: claim counts do not match filtered rows.")

        with st.expander("View full claim type summary table"):
            st.dataframe(claim_type_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Claim type analysis requires Claim_Type, Estimated_Claim_Amount, and Ultimate_Claim_Amount.")

    st.divider()

    # -------------------------
    # Categorical distributions
    # -------------------------
    st.subheader("📌 Distributions (Categorical Frequencies)")

    cat_cols = ["Weather_Condition", "Traffic_Condition", "Claim_Type", "Vehicle_Type"]
    cols = st.columns(2)
    plotted = 0

    for col in cat_cols:
        if col in filtered.columns:
            counts = filtered[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            fig = px.bar(counts, x=col, y="Count", title=f"{col} Frequency")
            cols[plotted % 2].plotly_chart(fig, use_container_width=True)
            plotted += 1

    st.divider()

    # -------------------------
    # Monthly Claims Trend
    # -------------------------
    st.subheader("📈 Monthly Claims Trend")

    if "Accident_Date" in filtered.columns:
        if not pd.api.types.is_datetime64_any_dtype(filtered["Accident_Date"]):
            filtered["Accident_Date"] = pd.to_datetime(filtered["Accident_Date"], errors="coerce")

        if filtered["Accident_Date"].notna().any():
            filtered["Accident_YearMonth"] = filtered["Accident_Date"].dt.to_period("M").astype(str)
            monthly_claims = filtered.groupby("Accident_YearMonth").size().reset_index(name="Claims")
            fig = px.line(monthly_claims, x="Accident_YearMonth", y="Claims", title="Monthly Claims Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Accident_Date exists but has no valid dates.")
    else:
        st.info("Accident_Date is missing so monthly trend cannot be generated.")

    st.divider()

    # -------------------------
    # Drivers of Ultimate Cost
    # -------------------------
    st.subheader("💸 Drivers of Ultimate Claim Cost")

    if "Ultimate_Claim_Amount" not in filtered.columns:
        st.info("Ultimate_Claim_Amount is missing so driver charts cannot be generated.")
        return

    for col in ["Claim_Type", "Weather_Condition", "Vehicle_Type"]:
        if col in filtered.columns:
            grp = filtered.groupby(col)["Ultimate_Claim_Amount"].mean().reset_index()
            fig = px.bar(grp, x=col, y="Ultimate_Claim_Amount", title=f"Avg Ultimate Claim Amount by {col}")
            st.plotly_chart(fig, use_container_width=True)
