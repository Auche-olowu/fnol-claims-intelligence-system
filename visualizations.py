import streamlit as st
import pandas as pd
import plotly.express as px

from models import add_features, fill_missing


def _safe_pct(n: float, d: float) -> float:
    return 0.0 if d == 0 else (n / d) * 100


def _insight_driver(filtered: pd.DataFrame) -> str:
    total = len(filtered)
    if total == 0:
        return "No data available for the current filters."

    msg = []
    if "Driver_Age" in filtered.columns and filtered["Driver_Age"].notna().any():
        avg_age = filtered["Driver_Age"].mean()
        min_age = filtered["Driver_Age"].min()
        max_age = filtered["Driver_Age"].max()
        msg.append(f"Average driver age is {avg_age:.0f} years (range {min_age:.0f}–{max_age:.0f}).")

    if "License_Age" in filtered.columns and filtered["License_Age"].notna().any():
        avg_lic = filtered["License_Age"].mean()
        msg.append(f"Average licence tenure is {avg_lic:.0f} years.")

    return " ".join(msg) if msg else "Driver demographics are available, but key age fields are missing."


def _insight_amounts(filtered: pd.DataFrame) -> str:
    if {"Estimated_Claim_Amount", "Ultimate_Claim_Amount"}.issubset(filtered.columns) and len(filtered) > 0:
        est = filtered["Estimated_Claim_Amount"].mean()
        ult = filtered["Ultimate_Claim_Amount"].mean()
        diff = ult - est
        direction = "higher" if diff > 0 else "lower"
        return (
            f"On average, ultimate cost is £{abs(diff):,.0f} {direction} than the FNOL estimate "
            f"(Avg Est £{est:,.0f} vs Avg Ult £{ult:,.0f})."
        )
    return "Claim amount insight unavailable (missing Estimated_Claim_Amount or Ultimate_Claim_Amount)."


def _insight_category(filtered: pd.DataFrame) -> str:
    if "Claim_Type" in filtered.columns and "Ultimate_Claim_Amount" in filtered.columns and len(filtered) > 0:
        totals = filtered.groupby("Claim_Type")["Ultimate_Claim_Amount"].sum().sort_values(ascending=False)
        top_type = totals.index[0]
        top_total = totals.iloc[0]
        share = _safe_pct(top_total, totals.sum())
        return f"'{top_type}' drives the highest total ultimate cost: £{top_total:,.0f} ({share:.1f}% of total)."
    return "Category insight unavailable (missing Claim_Type or Ultimate_Claim_Amount)."


def _insight_trends(filtered: pd.DataFrame) -> str:
    if "Accident_Date" in filtered.columns and filtered["Accident_Date"].notna().any():
        tmp = filtered.copy()
        tmp["YM"] = tmp["Accident_Date"].dt.to_period("M").astype(str)
        monthly = tmp.groupby("YM").size()
        if len(monthly) >= 2:
            peak = monthly.idxmax()
            low = monthly.idxmin()
            return (
                f"Peak claim month is {peak} with {monthly.loc[peak]:,} claims; "
                f"lowest is {low} with {monthly.loc[low]:,} claims."
            )
        return "Only one month of valid accident data is available under current filters."
    return "Trend insight unavailable (Accident_Date missing or invalid)."


def _insight_more(filtered: pd.DataFrame) -> str:
    parts = []
    if {"Weekday_Accident", "Ultimate_Claim_Amount"}.issubset(filtered.columns) and len(filtered) > 0:
        wd = filtered.groupby("Weekday_Accident")["Ultimate_Claim_Amount"].mean()
        best = wd.idxmax()
        best_val = wd.loc[best]
        parts.append(f"Highest average ultimate cost occurs on weekday {best} (Avg £{best_val:,.0f}).")

    if {"Season_of_Year", "Ultimate_Claim_Amount"}.issubset(filtered.columns) and len(filtered) > 0:
        se = filtered.groupby("Season_of_Year")["Ultimate_Claim_Amount"].mean().sort_values(ascending=False)
        top_season = se.index[0]
        parts.append(f"Season with highest average cost is {top_season} (Avg £{se.iloc[0]:,.0f}).")

    return " ".join(parts) if parts else "Additional insight unavailable (weekday/season fields missing)."


def render_visualizations(df: pd.DataFrame):
    st.title("📊 Visual Analytics")
    st.caption("Explore key patterns in FNOL claims using simple, business-friendly views.")

    if df is None or df.empty:
        st.warning("No data available.")
        return

    # merged_df already exists: just feature engineer + fill
    df = add_features(df)
    df = fill_missing(df, report=False)

    # Ensure datetime for time trends
    if "Accident_Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Accident_Date"]):
        df["Accident_Date"] = pd.to_datetime(df["Accident_Date"], errors="coerce")
    if "Settlement_Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Settlement_Date"]):
        df["Settlement_Date"] = pd.to_datetime(df["Settlement_Date"], errors="coerce")

    # -------------------------
    # Sidebar Filters
    # -------------------------
    with st.sidebar:
        st.header("🔎 Filters")

        if "Claim_Type" in df.columns:
            claim_type_options = ["All"] + sorted(df["Claim_Type"].dropna().unique().tolist())
            selected_claim_type = st.selectbox("Claim Type", claim_type_options, index=0)
        else:
            selected_claim_type = "All"

        if "Vehicle_Type" in df.columns:
            vehicle_type_options = ["All"] + sorted(df["Vehicle_Type"].dropna().unique().tolist())
            selected_vehicle_type = st.selectbox("Vehicle Type", vehicle_type_options, index=0)
        else:
            selected_vehicle_type = "All"

        if "Weather_Condition" in df.columns:
            weather_options = ["All"] + sorted(df["Weather_Condition"].dropna().unique().tolist())
            selected_weather = st.selectbox("Weather", weather_options, index=0)
        else:
            selected_weather = "All"

        if "Traffic_Condition" in df.columns:
            traffic_options = ["All"] + sorted(df["Traffic_Condition"].dropna().unique().tolist())
            selected_traffic = st.selectbox("Traffic", traffic_options, index=0)
        else:
            selected_traffic = "All"

        date_range = None
        if "Accident_Date" in df.columns and df["Accident_Date"].notna().any():
            min_d = df["Accident_Date"].min().date()
            max_d = df["Accident_Date"].max().date()
            date_range = st.date_input("Accident date range", value=(min_d, max_d))

    # Apply filters
    filtered = df.copy()

    if selected_claim_type != "All" and "Claim_Type" in filtered.columns:
        filtered = filtered[filtered["Claim_Type"] == selected_claim_type]

    if selected_vehicle_type != "All" and "Vehicle_Type" in filtered.columns:
        filtered = filtered[filtered["Vehicle_Type"] == selected_vehicle_type]

    if selected_weather != "All" and "Weather_Condition" in filtered.columns:
        filtered = filtered[filtered["Weather_Condition"] == selected_weather]

    if selected_traffic != "All" and "Traffic_Condition" in filtered.columns:
        filtered = filtered[filtered["Traffic_Condition"] == selected_traffic]

    if date_range and "Accident_Date" in filtered.columns and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["Accident_Date"].dt.date >= start) &
            (filtered["Accident_Date"].dt.date <= end)
        ]

    if filtered.empty:
        st.warning("No records match your filters. Try selecting 'All' for some filters.")
        return

    # -------------------------
    # KPI strip (safe)
    # -------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(filtered):,}")

    if "Ultimate_Claim_Amount" in filtered.columns:
        k2.metric("Avg Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].mean():,.0f}")
        k3.metric("Min Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].min():,.0f}")
        k4.metric("Max Ultimate (£)", f"{filtered['Ultimate_Claim_Amount'].max():,.0f}")
    else:
        k2.metric("Avg Ultimate (£)", "N/A")
        k3.metric("Min Ultimate (£)", "N/A")
        k4.metric("Max Ultimate (£)", "N/A")

    st.divider()

    # -------------------------
    # View selector
    # -------------------------
    view = st.selectbox(
        "Select analysis view",
        [
            "Driver demographics",
            "Claim amount analysis",
            "Category analysis",
            "Monthly trend analysis",
            "More (Weekday / Season)"
        ],
        index=0
    )

    # VIEW 1
    if view == "Driver demographics":
        st.subheader("👤 Driver demographics")

        c1, c2 = st.columns(2)
        if "Driver_Age" in filtered.columns:
            c1.plotly_chart(
                px.histogram(filtered, x="Driver_Age", nbins=30, title="Driver Age Distribution"),
                use_container_width=True
            )
        if "License_Age" in filtered.columns:
            c2.plotly_chart(
                px.histogram(filtered, x="License_Age", nbins=30, title="License Age Distribution"),
                use_container_width=True
            )

        if {"Driver_Age", "Ultimate_Claim_Amount"}.issubset(filtered.columns):
            st.plotly_chart(
                px.scatter(filtered, x="Driver_Age", y="Ultimate_Claim_Amount",
                           title="Ultimate Claim Amount vs Driver Age", opacity=0.5),
                use_container_width=True
            )

        st.info(f"Insight: {_insight_driver(filtered)}")

    # VIEW 2
    elif view == "Claim amount analysis":
        st.subheader("💷 Claim amount analysis")

        if "Estimated_Claim_Amount" in filtered.columns:
            st.plotly_chart(
                px.histogram(filtered, x="Estimated_Claim_Amount", nbins=40,
                             title="Estimated Claim Amount Distribution"),
                use_container_width=True
            )
        if "Ultimate_Claim_Amount" in filtered.columns:
            st.plotly_chart(
                px.histogram(filtered, x="Ultimate_Claim_Amount", nbins=40,
                             title="Ultimate Claim Amount Distribution"),
                use_container_width=True
            )
        if {"Estimated_Claim_Amount", "Ultimate_Claim_Amount"}.issubset(filtered.columns):
            st.plotly_chart(
                px.scatter(filtered, x="Estimated_Claim_Amount", y="Ultimate_Claim_Amount",
                           title="Estimated vs Ultimate (FNOL estimate accuracy)", opacity=0.5),
                use_container_width=True
            )

        st.info(f"Insight: {_insight_amounts(filtered)}")

    # VIEW 3
    elif view == "Category analysis":
        st.subheader("🏷️ Category analysis")

        c1, c2 = st.columns(2)
        for i, col in enumerate(["Claim_Type", "Vehicle_Type", "Weather_Condition", "Traffic_Condition"]):
            if col in filtered.columns:
                counts = filtered[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                fig = px.bar(counts, x=col, y="Count", title=f"{col} Frequency")
                (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

        if "Ultimate_Claim_Amount" in filtered.columns:
            st.markdown("### Average Ultimate by category")
            for col in ["Claim_Type", "Vehicle_Type", "Weather_Condition"]:
                if col in filtered.columns:
                    grp = filtered.groupby(col)["Ultimate_Claim_Amount"].mean().reset_index()
                    st.plotly_chart(
                        px.bar(grp, x=col, y="Ultimate_Claim_Amount",
                               title=f"Avg Ultimate Claim Amount by {col}"),
                        use_container_width=True
                    )

        st.info(f"Insight: {_insight_category(filtered)}")

    # VIEW 4
    elif view == "Monthly trend analysis":
        st.subheader("📈 Monthly trend analysis")

        if "Accident_Date" in filtered.columns and filtered["Accident_Date"].notna().any():
            tmp = filtered.copy()
            tmp["Accident_YearMonth"] = tmp["Accident_Date"].dt.to_period("M").astype(str)
            monthly_claims = tmp.groupby("Accident_YearMonth").size().reset_index(name="Claims")
            st.plotly_chart(
                px.line(monthly_claims, x="Accident_YearMonth", y="Claims", title="Monthly Claims Count"),
                use_container_width=True
            )
        else:
            st.info("Accident_Date missing/invalid; cannot build claims trend.")

        if "Settlement_Date" in filtered.columns and filtered["Settlement_Date"].notna().any():
            tmp = filtered.copy()
            tmp["Settlement_YearMonth"] = tmp["Settlement_Date"].dt.to_period("M").astype(str)
            monthly_settlements = tmp.groupby("Settlement_YearMonth").size().reset_index(name="Settlements")
            st.plotly_chart(
                px.line(monthly_settlements, x="Settlement_YearMonth", y="Settlements",
                        title="Monthly Settlements Count"),
                use_container_width=True
            )

        st.info(f"Insight: {_insight_trends(filtered)}")

    # VIEW 5
    else:
        st.subheader("➕ Additional insights")

        if {"Weekday_Accident", "Ultimate_Claim_Amount"}.issubset(filtered.columns):
            weekday_avg = filtered.groupby("Weekday_Accident")["Ultimate_Claim_Amount"].mean().reset_index()
            st.plotly_chart(
                px.bar(weekday_avg, x="Weekday_Accident", y="Ultimate_Claim_Amount",
                       title="Avg Ultimate Claim Amount by Weekday"),
                use_container_width=True
            )

        if {"Season_of_Year", "Ultimate_Claim_Amount"}.issubset(filtered.columns):
            season_avg = filtered.groupby("Season_of_Year")["Ultimate_Claim_Amount"].mean().reset_index()
            st.plotly_chart(
                px.bar(season_avg, x="Season_of_Year", y="Ultimate_Claim_Amount",
                       title="Avg Ultimate Claim Amount by Season"),
                use_container_width=True
            )

        st.info(f"Insight: {_insight_more(filtered)}")
