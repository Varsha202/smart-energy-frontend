
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Smart Campus Energy Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Driven Smart Campus Energy Analytics")
st.caption("Monitor • Analyze • Forecast • Optimize")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data_dir = Path("data")
    files = list(data_dir.rglob("*.csv")) if data_dir.exists() else []

    for file in files:
        try:
            df = pd.read_csv(file)
            if len(df.columns) >= 2 and not df.empty:
                return df, file.name
        except Exception:
            continue

    # Clearly labelled demo data when no usable CSV exists.
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90)
    consumption = (
        450 + np.arange(90) * 0.8
        + 35 * np.sin(np.arange(90) * 2 * np.pi / 7)
        + rng.normal(0, 18, 90)
    )
    return pd.DataFrame({
        "Date": dates,
        "Energy Consumption": np.maximum(consumption, 0)
    }), "Generated demo data"

df, data_source = load_data()

st.info(f"Data source: {data_source}")

# Allow the user to load a different dataset.
uploaded = st.sidebar.file_uploader("Upload energy CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    data_source = uploaded.name
    st.sidebar.success("CSV loaded")

# ---------------- PREPROCESSING ----------------
df.columns = [str(c).strip() for c in df.columns]

with st.sidebar:
    st.header("Data configuration")
    st.caption("Choose the columns matching your dataset.")

    date_options = [
        c for c in df.columns
        if any(k in c.lower() for k in ["date", "time", "day"])
    ]
    date_options = ["None"] + date_options
    date_col = st.selectbox("Date column", date_options)

    numeric_options = df.select_dtypes(include=np.number).columns.tolist()
    preferred = [
        c for c in numeric_options
        if any(k in c.lower() for k in [
            "consumption", "energy", "kwh", "total", "electricity", "usage"
        ])
    ]
    if not numeric_options:
        st.error("The CSV needs at least one numeric energy column.")
        st.stop()

    default_energy = preferred[0] if preferred else numeric_options[0]
    energy_col = st.selectbox(
        "Energy consumption column",
        numeric_options,
        index=numeric_options.index(default_energy)
    )

    tariff = st.number_input(
        "Electricity tariff (₹ per unit)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )
    emission_factor = st.number_input(
        "Emission factor (kg CO₂ per unit)",
        min_value=0.0,
        value=0.716,
        step=0.01
    )

df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")

if date_col != "None":
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
else:
    df["_date"] = pd.NaT

df = df.dropna(subset=[energy_col]).copy()
df = df[df[energy_col] >= 0]

if df.empty:
    st.error("No valid non-negative energy values were found.")
    st.stop()

# Use row order if no usable dates are available.
if df["_date"].notna().any():
    df = df.dropna(subset=["_date"]).sort_values("_date")
else:
    df["_date"] = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=len(df),
        freq="D"
    )
    st.warning(
        "No valid dates were found. Dates below use row order, "
        "so they are not historical dates from your dataset."
    )

if df.empty:
    st.error("No usable rows remain after date and energy validation.")
    st.stop()

# ---------------- SIDEBAR FILTER ----------------
min_date = df["_date"].min().date()
max_date = df["_date"].max().date()

with st.sidebar:
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = df[
        df["_date"].dt.date.between(start_date, end_date)
    ].copy()
else:
    filtered = df.copy()

if filtered.empty:
    st.warning("No records match this date range.")
    st.stop()

# Aggregate by day to avoid treating multiple readings as separate days.
daily = (
    filtered.set_index("_date")[energy_col]
    .resample("D").sum()
    .dropna()
    .rename("Energy")
    .reset_index()
    .rename(columns={"_date": "Date"})
)

# ---------------- KPI CARDS ----------------
total = filtered[energy_col].sum()
average = daily["Energy"].mean()
peak = daily["Energy"].max()
estimated_cost = total * tariff
estimated_emissions = total * emission_factor

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total consumption", f"{total:,.2f} units")
c2.metric("Daily average", f"{average:,.2f} units")
c3.metric("Peak daily usage", f"{peak:,.2f} units")
c4.metric("Estimated electricity cost", f"₹{estimated_cost:,.2f}")

c5, c6 = st.columns(2)
c5.metric("Estimated carbon emissions", f"{estimated_emissions:,.2f} kg CO₂")
c6.metric("Days analyzed", f"{len(daily)}")

st.caption(
    "Cost and emissions are estimates based on the tariff and emission "
    "factor you selected. Confirm the units and factors for your campus."
)

# ---------------- TABS ----------------
overview, forecast_tab, anomaly_tab, insights_tab, data_tab = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "🚨 Anomalies",
    "🌱 Recommendations",
    "📁 Data"
])

with overview:
    st.subheader("Daily energy consumption")
    fig = px.area(
        daily, x="Date", y="Energy",
        title="Daily consumption trend",
        labels={"Energy": "Energy (dataset units)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily consumption distribution")
    hist = px.histogram(daily, x="Energy", nbins=20)
    st.plotly_chart(hist, use_container_width=True)

with forecast_tab:
    st.subheader("Next three months — experimental forecast")
    st.caption(
        "This is a basic linear-trend model, not a validated seasonal forecast. "
        "Its reliability depends on the quantity and quality of historical data."
    )

    horizon = 90
    if len(daily) < 14:
        st.warning(
            "At least 14 daily observations are required for this demonstration "
            "forecast. Collect more data before relying on predictions."
        )
    else:
        daily["TimeIndex"] = np.arange(len(daily))
        X = daily[["TimeIndex"]]
        y = daily["Energy"].values

        model = LinearRegression()
        model.fit(X, y)

        future_idx = np.arange(len(daily), len(daily) + horizon)
        predictions = model.predict(future_idx.reshape(-1, 1))
        predictions = np.maximum(predictions, 0)

        future_dates = pd.date_range(
            daily["Date"].max() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        future = pd.DataFrame({
            "Date": future_dates,
            "Predicted energy": predictions
        })

        fig = px.line(
            future, x="Date", y="Predicted energy",
            title="90-day projected energy consumption"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(future, use_container_width=True)

        st.download_button(
            "Download forecast CSV",
            future.to_csv(index=False).encode("utf-8"),
            file_name="energy_forecast.csv",
            mime="text/csv"
        )

with anomaly_tab:
    st.subheader("Unusually high or low consumption")

    if len(daily) < 10:
        st.info("At least 10 daily observations are required for anomaly analysis.")
    else:
        contamination = st.slider(
            "Expected fraction of unusual days",
            min_value=0.01,
            max_value=0.15,
            value=0.05,
            step=0.01
        )

        detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        daily["Anomaly"] = detector.fit_predict(daily[["Energy"]])
        daily["Status"] = np.where(
            daily["Anomaly"] == -1, "Unusual", "Normal"
        )

        fig = px.scatter(
            daily, x="Date", y="Energy", color="Status",
            title="Daily consumption anomaly screening"
        )
        st.plotly_chart(fig, use_container_width=True)

        unusual = daily[daily["Anomaly"] == -1]
        st.write(f"Unusual observations detected: {len(unusual)}")
        st.dataframe(unusual[["Date", "Energy", "Status"]], use_container_width=True)
        st.caption(
            "Unusual consumption does not necessarily mean an equipment fault."
        )

with insights_tab:
    st.subheader("Energy-saving recommendations")

    weekday_avg = daily.loc[
        daily["Date"].dt.dayofweek < 5, "Energy"
    ].mean()
    weekend_avg = daily.loc[
        daily["Date"].dt.dayofweek >= 5, "Energy"
    ].mean()

    st.write("**Recommendations to investigate:**")
    st.markdown("""
    - Review air-conditioning schedules in unoccupied classrooms.
    - Switch off lighting and equipment when rooms are unused.
    - Investigate unusually high consumption days.
    - Compare usage across buildings if facility-level data becomes available.
    - Track consumption regularly and evaluate changes after interventions.
    """)

    if pd.notna(weekday_avg) and pd.notna(weekend_avg):
        st.write(f"Average weekday consumption: **{weekday_avg:.2f} units/day**")
        st.write(f"Average weekend consumption: **{weekend_avg:.2f} units/day**")
        st.caption(
            "Differences may reflect campus schedules; they do not prove waste."
        )

    st.info(
        "No guaranteed savings are claimed. Verify recommendations using "
        "campus schedules, actual tariffs and follow-up measurements."
    )

with data_tab:
    st.subheader("Records used in this dashboard")
    st.write(f"Source: {data_source}")
    st.dataframe(filtered.drop(columns=["_date"], errors="ignore"),
                 use_container_width=True)

    st.download_button(
        "Download filtered data",
        filtered.drop(columns=["_date"], errors="ignore")
        .to_csv(index=False).encode("utf-8"),
        file_name="energy_data.csv",
        mime="text/csv"
    )
