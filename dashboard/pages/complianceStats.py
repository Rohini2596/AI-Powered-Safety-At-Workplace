import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------- CONFIG --------
LOG_PATH = r"C:\WE\Infosys\dashboard\data\violations.csv"
CHART_THEME = "plotly_dark"
TIME_GROUP = "hour"               # hour / day / minute


# -------- SAFE LOADER --------
@st.cache_data
def load_logs():
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as e:
        st.error(f"Failed to load violations.csv: {e}")
        return pd.DataFrame()

    # Validate required columns
    required_cols = ["timestamp", "worker_id", "violation_type", "confidence", "status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        return pd.DataFrame()

    # Convert timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except:
        st.error("Timestamp format error — must be datetime.")
        return pd.DataFrame()

    return df


# -------- METRICS --------
def compute_metrics(df):
    if df.empty:
        return 0, 0, 0, 0

    total = len(df)
    compliant = len(df[df["status"] == "Safe"])
    violations = len(df[df["status"] == "Violation"])
    compliance_percent = (compliant / total * 100) if total else 0
    return total, compliant, violations, compliance_percent


# -------- TIME GROUPING --------
def trend_data(df, group="hour"):
    if df.empty:
        return pd.DataFrame({"time_group": [], "violations": []})

    if group == "hour":
        df["time_group"] = df["timestamp"].dt.hour
    elif group == "day":
        df["time_group"] = df["timestamp"].dt.date
    elif group == "minute":
        df["time_group"] = df["timestamp"].dt.strftime("%H:%M")

    trend = (
        df.groupby("time_group")["status"]
        .apply(lambda x: (x == "Violation").sum())
        .reset_index(name="violations")
    )

    return trend


# -------- MAIN UI FUNCTION --------
def render_dashboard():

    st.title("📊 PPE Compliance Statistics")
    st.markdown("Real-time and historical analytics from violation logs.")

    df = load_logs()

    if df.empty:
        st.warning("No data available to display.")
        return

    # -------------------- FILTERS --------------------
    st.sidebar.header("🔍 Filters")

    start_date = st.sidebar.date_input("Start Date", df["timestamp"].min().date())
    end_date = st.sidebar.date_input("End Date", df["timestamp"].max().date())

    # Fix: protect multiselect from empty dataframe issues
    violation_options = df["violation_type"].dropna().unique().tolist()

    violation_types = st.sidebar.multiselect(
        "Violation Types",
        options=violation_options,
        default=violation_options
    )

    df_filtered = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date) &
        (df["violation_type"].isin(violation_types))
    ]

    # -------------------- KPI CARDS --------------------
    total, compliant, violations, compliance_percent = compute_metrics(df_filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", total)
    col2.metric("Compliant", compliant)
    col3.metric("Violations", violations)
    col4.metric("Compliance %", f"{compliance_percent:.2f}%")

    st.markdown("---")

    # -------------------- GAUGE CHART --------------------
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=compliance_percent,
            title={'text': "Compliance Percentage"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'green'}
                ]
            }
        )
    )
    fig_gauge.update_layout(template=CHART_THEME)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------- TREND CHART --------------------
    trend = trend_data(df_filtered, group=TIME_GROUP)

    st.subheader("📈 Violations Over Time")

    if not trend.empty:
        fig_line = px.line(
            trend,
            x="time_group",
            y="violations",
            title="Violation Trend",
            markers=True,
            labels={"time_group": "Time", "violations": "Violations"},
            template=CHART_THEME
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No trend data available for the selected filters.")

    # -------------------- VIOLATION PIE --------------------
    st.subheader("📊 Violation Type Distribution")

    if len(df_filtered) > 0:
        type_counts = df_filtered["violation_type"].value_counts().reset_index()
        type_counts.columns = ["violation_type", "count"]

        fig_pie = px.pie(
            type_counts,
            values="count",
            names="violation_type",
            title="Violation Breakdown",
            template=CHART_THEME
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No violation type data available.")

    # -------------------- WORKER COMPLIANCE --------------------
    st.subheader("👷 Worker Compliance Summary")

    if len(df_filtered) > 0:
        worker_summary = (
            df_filtered.groupby("worker_id")["status"]
            .apply(lambda x: (x == "Safe").sum() / len(x) * 100)
            .reset_index(name="compliance_percent")
        )

        fig_bar = px.bar(
            worker_summary,
            x="compliance_percent",
            y="worker_id",
            orientation="h",
            title="Worker-wise Compliance %",
            labels={"compliance_percent": "Compliance %", "worker_id": "Worker"},
            template=CHART_THEME
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No worker compliance data available.")

    st.success("✔ Compliance Stats Module Loaded Successfully")
