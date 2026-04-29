import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="SmartRain Dashboard: From Forecast to Flood Readiness", layout="wide")

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
:root {
    --bg-main: #eef5f5;
    --sidebar-top: #0b4f6c;
    --sidebar-bottom: #167b80;
    --deep-blue: #12355b;
    --teal: #1b9c85;
    --blue: #1565c0;
    --mint: #dff4f1;
    --card-bg: rgba(255,255,255,0.55);
    --card-border: rgba(255,255,255,0.65);
    --shadow: 0 10px 28px rgba(18,53,91,0.12);
    --table-header: #0b4f6c;
    --gold: #f2b134;
    --danger: #e85d75;
}

.stApp {
    background: radial-gradient(circle at top left, #f7fcfc 0%, #eef5f5 45%, #e9f1f1 100%);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    padding-left: 2.6rem;
    padding-right: 2.6rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-top), var(--sidebar-bottom));
    width: 360px;
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 4px 0 18px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p {
    color: white !important;
    font-size: 20px;
    font-weight: 700;
}

section[data-testid="stSidebar"] h2 {
    font-size: 28px;
    font-weight: 900;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 14px;
    border: 2px solid #c8e7e1;
    min-height: 54px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #000000 !important;
    font-weight: 800;
    font-size: 22px;
    text-align: center;
    width: 100%;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-align: center;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #000000 !important;
}

div[role="listbox"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #d7ece8;
}

div[role="option"] {
    color: #000000 !important;
    font-weight: 700;
    font-size: 20px;
}

div[role="option"]:hover {
    background-color: #e6f7f4;
}

section[data-testid="stSidebar"] div[data-baseweb="slider"] span {
    font-size: 22px;
    font-weight: 800;
}

h1 {
    color: var(--deep-blue);
    font-size: 48px;
    font-weight: 900;
    letter-spacing: 0.2px;
}

h2 {
    color: var(--deep-blue);
    font-size: 32px;
    font-weight: 800;
}

h3 {
    color: var(--deep-blue);
    font-size: 26px;
    font-weight: 800;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(223,244,241,0.70));
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 24px;
    padding: 22px 28px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(8px);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 40px;
    font-weight: 900;
    color: var(--deep-blue);
    margin-bottom: 4px;
}

.hero-subtitle {
    font-size: 18px;
    font-weight: 600;
    color: #4e6276;
}

.hero-badges {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.hero-badge {
    background: rgba(27,156,133,0.12);
    color: var(--deep-blue);
    border: 1px solid rgba(27,156,133,0.22);
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 800;
}

.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 24px;
    padding: 16px 18px 10px 18px;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow);
    margin-bottom: 14px;
}

.glass-card iframe {
    border-radius: 18px !important;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 8px;
    margin-bottom: 20px;
}

.info-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(232,247,245,0.82));
    border: 1px solid rgba(255,255,255,0.85);
    border-radius: 22px;
    padding: 18px;
    box-shadow: var(--shadow);
    min-height: 126px;
}

.info-card-title {
    font-size: 16px;
    font-weight: 900;
    color: var(--deep-blue);
    margin-bottom: 8px;
}

.info-card-text {
    font-size: 15px;
    font-weight: 600;
    color: #4f5f6d;
    line-height: 1.55;
}

button[data-baseweb="tab"] {
    background-color: rgba(223,244,241,0.92);
    border: 2px solid #bde4dd;
    border-radius: 16px;
    font-size: 24px;
    font-weight: 900;
    padding: 12px 24px;
    margin-right: 10px;
    width: auto;
    min-width: fit-content;
    color: var(--deep-blue);
    box-shadow: 0 5px 12px rgba(18,53,91,0.06);
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1b9c85, #24a894);
    color: white;
    border: 2px solid #1b9c85;
    box-shadow: 0 8px 16px rgba(27,156,133,0.24);
}

button[data-baseweb="tab"]:hover {
    background-color: #d3f0eb;
}

div[data-baseweb="tab-list"] {
    gap: 12px;
}

.circle-metric-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 42px;
    margin-top: 24px;
    margin-bottom: 34px;
    flex-wrap: wrap;
}

.circle-metric {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    background: radial-gradient(circle at top, #ffffff 0%, #edf9f7 62%, #d6f0ea 100%);
    position: relative;
    box-shadow: 0 10px 26px rgba(18,53,91,0.12), 0 18px 30px rgba(18,53,91,0.10);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: all 0.30s ease;
}

.circle-metric::before {
    content: "";
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1b9c85, #3fb6c6, #7ed9cf);
    z-index: -1;
}

.circle-metric:hover {
    transform: translateY(-6px) scale(1.05);
    box-shadow: 0 18px 36px rgba(18,53,91,0.16), 0 24px 42px rgba(18,53,91,0.18);
}

.circle-metric-label {
    font-size: 24px;
    font-weight: 800;
    color: var(--deep-blue);
    margin-bottom: 10px;
    text-align: center;
    letter-spacing: 0.8px;
}

.circle-metric-value {
    font-size: 38px;
    font-weight: 900;
    color: var(--sidebar-top);
    text-align: center;
    line-height: 1.05;
    padding: 0 12px;
}

.risk-box {
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: var(--shadow);
    margin-top: 4px;
    margin-bottom: 16px;
    font-weight: 700;
}

.risk-low {
    background: linear-gradient(135deg, rgba(223,244,241,0.95), rgba(232,255,246,0.95));
    border: 1px solid rgba(27,156,133,0.22);
    color: #0e5f56;
}

.risk-medium {
    background: linear-gradient(135deg, rgba(255,244,224,0.98), rgba(255,250,235,0.98));
    border: 1px solid rgba(242,177,52,0.35);
    color: #8a5a00;
}

.risk-high {
    background: linear-gradient(135deg, rgba(255,229,233,0.98), rgba(255,244,246,0.98));
    border: 1px solid rgba(232,93,117,0.35);
    color: #9a213a;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #d5ece8;
    box-shadow: 0 6px 14px rgba(18,53,91,0.06);
}

thead tr th {
    background-color: var(--table-header);
    color: white;
    font-size: 16px;
    font-weight: 800;
    text-align: center;
}

tbody tr td {
    font-size: 15px;
    font-weight: 600;
    color: var(--deep-blue);
    text-align: center;
}

tbody tr:nth-child(even) {
    background-color: #eefaf8;
}

.stButton > button {
    background: linear-gradient(135deg, #1b9c85, #23a894);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
    border: none;
    box-shadow: 0 8px 16px rgba(27,156,133,0.20);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #1565c0, #2b7bd6);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
    border: none;
    box-shadow: 0 8px 16px rgba(21,101,192,0.20);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    font-size: 17px;
    font-weight: 600;
}

[data-testid="stCaptionContainer"] {
    font-size: 15px;
    color: #5f6b7a;
}

@media (max-width: 900px) {
    .info-grid { grid-template-columns: 1fr; }
    button[data-baseweb="tab"] { font-size: 18px; padding: 10px 16px; }
    .circle-metric { width: 150px; height: 150px; }
    .circle-metric-label { font-size: 14px; }
    .circle-metric-value { font-size: 28px; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS
# =========================
AREA_OPTIONS = ["Alor Setar", "Kuantan", "Subang"]


def get_model_insight(area_name, model_name, mape_value):
    if np.isnan(mape_value):
        quality = "performance could not be evaluated reliably"
    elif mape_value < 20:
        quality = "strong forecasting performance"
    elif mape_value < 40:
        quality = "acceptable forecasting performance"
    else:
        quality = "moderate forecasting performance with room for improvement"

    if model_name == "STL-ARIMA":
        model_note = "The selected model captures seasonal rainfall structure before ARIMA-based forecasting."
    else:
        model_note = "The selected model directly models temporal dependence and seasonal rainfall behaviour."

    return f"For {area_name}, the current results indicate {quality}. {model_note}"




def get_risk_level(forecast_series, historical_series):
    q75 = historical_series.quantile(0.75)
    q90 = historical_series.quantile(0.90)

    max_forecast = float(np.nanmax(forecast_series))
    mean_forecast = float(np.nanmean(forecast_series))

    risk_score = max(max_forecast, mean_forecast)

    if risk_score >= q90:
        return "High", "risk-high", "🔴 High rainfall risk detected based on historical threshold."
    elif risk_score >= q75:
        return "Moderate", "risk-medium", "🟠 Moderate rainfall risk detected based on historical threshold."
    return "Low", "risk-low", "🟢 Low rainfall risk detected based on historical threshold."

def get_best_model_name(sarima_mape_value, stl_mape_value):
    if np.isnan(sarima_mape_value) and np.isnan(stl_mape_value):
        return "Not available"
    if np.isnan(sarima_mape_value):
        return "STL-ARIMA"
    if np.isnan(stl_mape_value):
        return "SARIMA"
    return "SARIMA" if sarima_mape_value < stl_mape_value else "STL-ARIMA"


def get_station_map_data():
    return pd.DataFrame({
        "Area": ["Alor Setar", "Kuantan", "Subang"],
        "Latitude": [6.1248, 3.8077, 3.1300],
        "Longitude": [100.3678, 103.3260, 101.5490],
        "Region": ["Kedah", "Pahang", "Selangor"]
    })


def get_file_by_area(area_name):
    if area_name == "Alor Setar":
        return "data1_Alor Setar.csv"
    elif area_name == "Kuantan":
        return "data3_Kuantan.csv"
    return "data2.csv"


@st.cache_data
def load_data(file_path, area_name):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    if "Rainfall" not in data.columns:
        rainfall_candidates = [col for col in data.columns if "rain" in col.lower()]
        if rainfall_candidates:
            data.rename(columns={rainfall_candidates[0]: "Rainfall"}, inplace=True)

    if area_name in ["Alor Setar", "Kuantan"]:
        data["Date"] = pd.to_datetime(data["Date"], format="%y-%b", errors="coerce")
    else:
        data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    data["Rainfall"] = pd.to_numeric(data["Rainfall"], errors="coerce")
    data = data.dropna(subset=["Date", "Rainfall"]).copy()
    data = data.sort_values("Date")
    data.set_index("Date", inplace=True)

    data = data.asfreq("MS")
    data["Rainfall"] = data["Rainfall"].interpolate(method="linear").ffill().bfill()

    return data


@st.cache_data
def load_all_areas():
    alor = load_data("data1_Alor Setar.csv", "Alor Setar").rename(columns={"Rainfall": "Alor Setar"})
    subang = load_data("data2.csv", "Subang").rename(columns={"Rainfall": "Subang"})
    kuantan = load_data("data3_Kuantan.csv", "Kuantan").rename(columns={"Rainfall": "Kuantan"})
    return pd.concat([alor["Alor Setar"], subang["Subang"], kuantan["Kuantan"]], axis=1)


def fit_sarima(train_series, steps, area_name):
    if area_name == "Alor Setar":
        order = (2, 1, 0)
        seasonal_order = (1, 1, 0, 12)
    elif area_name == "Kuantan":
        order = (3, 1, 0)
        seasonal_order = (1, 1, 0, 12)
    else:
        order = (1, 0, 1)
        seasonal_order = (3, 1, 0, 12)

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    forecast_obj = result.get_forecast(steps=steps)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)

    forecast_index = pd.date_range(
        start=train_series.index[-1] + pd.offsets.MonthBegin(1),
        periods=steps,
        freq="MS"
    )
    forecast.index = forecast_index
    conf_int.index = forecast_index
    conf_int.columns = ["Lower 95% CI", "Upper 95% CI"]
    return result, forecast, conf_int


def fit_stl_arima(train_series, steps, area_name):
    stl = STL(train_series, period=12, robust=True)
    stl_result = stl.fit()

    adjusted_series = train_series - stl_result.seasonal

    if area_name == "Alor Setar":
        order = (2, 2, 0)
    elif area_name == "Kuantan":
        order = (0, 0, 0)
    else:
        order = (2, 1, 1)

    model = SARIMAX(
        adjusted_series,
        order=order,
        seasonal_order=(0, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    forecast_obj = result.get_forecast(steps=steps)
    adjusted_forecast = forecast_obj.predicted_mean
    adjusted_ci = forecast_obj.conf_int(alpha=0.05)

    seasonal_pattern = stl_result.seasonal[-12:].values
    repeated_seasonal = np.resize(seasonal_pattern, steps)

    forecast_index = pd.date_range(
        start=train_series.index[-1] + pd.offsets.MonthBegin(1),
        periods=steps,
        freq="MS"
    )

    final_forecast = pd.Series(adjusted_forecast.values + repeated_seasonal, index=forecast_index)
    final_ci = adjusted_ci.copy()
    final_ci.index = forecast_index
    final_ci.iloc[:, 0] = final_ci.iloc[:, 0].values + repeated_seasonal
    final_ci.iloc[:, 1] = final_ci.iloc[:, 1].values + repeated_seasonal
    final_ci.columns = ["Lower 95% CI", "Upper 95% CI"]

    return stl_result, result, final_forecast, final_ci


def calculate_metrics(actual_series, predicted_series):
    common_index = actual_series.index.intersection(predicted_series.index)
    actual_aligned = actual_series.loc[common_index]
    predicted_aligned = predicted_series.loc[common_index]

    metrics_df = pd.DataFrame({
        "Actual": actual_aligned,
        "Predicted": predicted_aligned
    }).dropna()

    if not metrics_df.empty:
        mae = mean_absolute_error(metrics_df["Actual"], metrics_df["Predicted"])
        rmse = np.sqrt(mean_squared_error(metrics_df["Actual"], metrics_df["Predicted"]))
        denominator = metrics_df["Actual"].replace(0, np.nan)
        mape = np.nanmean(np.abs((metrics_df["Actual"] - metrics_df["Predicted"]) / denominator)) * 100
    else:
        mae = np.nan
        rmse = np.nan
        mape = np.nan

    return metrics_df, mae, rmse, mape

# =========================
# TITLE
# =========================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌧️ SmartRain Dashboard: From Forecast to Flood Readiness</div>
    <div class="hero-subtitle">Turning rainfall forecasts into early warning insights for smarter flood preparedness across Malaysia</div>
    <div class="hero-badges">
        <div class="hero-badge">STL-ARIMA</div>
        <div class="hero-badge">SARIMA</div>
        <div class="hero-badge">Forecast Accuracy</div>
        <div class="hero-badge">Decision Support</div>
        <div class="hero-badge">Interactive Malaysia Map</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "selected_area" not in st.session_state:
    st.session_state.selected_area = "Kuantan"

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Forecast Settings")

area = st.sidebar.selectbox(
    "Select Area",
    AREA_OPTIONS,
    index=AREA_OPTIONS.index(st.session_state.selected_area)
)
st.session_state.selected_area = area

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["STL-ARIMA", "SARIMA"]
)

forecast_period = st.sidebar.slider(
    "Forecast Months",
    min_value=3,
    max_value=24,
    value=12,
    step=1
)

test_size = st.sidebar.slider(
    "Testing Months",
    min_value=6,
    max_value=24,
    value=12,
    step=1
)

file = get_file_by_area(area)

# =========================
# MAP SECTION
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Interactive Malaysia Rainfall Station Map")

map_df = get_station_map_data()

malaysia_map = folium.Map(
    location=[4.5, 102.0],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True
)

bounds = []

for _, row in map_df.iterrows():
    selected = row["Area"] == area
    fill_color = "#1b9c85" if selected else "#1565c0"
    radius = 16 if selected else 11
    weight = 4 if selected else 2

    html_popup = f"""
    <div style="font-family: Arial; font-size: 14px; width: 180px;">
        <b>Area:</b> {row['Area']}<br>
        <b>Region:</b> {row['Region']}<br>
        <b>Status:</b> {"Selected station" if selected else "Click to select"}
    </div>
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=radius,
        color="white",
        weight=weight,
        fill=True,
        fill_color=fill_color,
        fill_opacity=0.92,
        tooltip=row["Area"],
        popup=folium.Popup(html_popup, max_width=220)
    ).add_to(malaysia_map)

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 13px;
                font-weight: 800;
                color: #12355b;
                white-space: nowrap;
                text-align: center;
                transform: translate(-50%, 16px);
            ">
                {row['Area']}
            </div>
            """
        )
    ).add_to(malaysia_map)

    bounds.append([row["Latitude"], row["Longitude"]])

if bounds:
    malaysia_map.fit_bounds(bounds, padding=(40, 40))

map_output = st_folium(
    malaysia_map,
    width=None,
    height=520,
    returned_objects=["last_object_clicked_tooltip"],
    key="malaysia_station_map"
)

clicked_area = map_output.get("last_object_clicked_tooltip")

if clicked_area in AREA_OPTIONS and clicked_area != st.session_state.selected_area:
    st.session_state.selected_area = clicked_area
    st.rerun()

st.caption(f"Click any station marker on the map to select the study area. Current selected area: {st.session_state.selected_area}.")
st.markdown('</div>', unsafe_allow_html=True)

area = st.session_state.selected_area
file = get_file_by_area(area)

# =========================
# FRONT DASHBOARD PLOT
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Monthly Rainfall Comparison for Alor Setar, Subang, and Kuantan")

all_areas_df = load_all_areas()

fig, ax = plt.subplots(figsize=(6.6, 3.1))
ax.plot(all_areas_df.index, all_areas_df["Alor Setar"], linewidth=1.0, label="Alor Setar")
ax.plot(all_areas_df.index, all_areas_df["Subang"], linewidth=1.0, label="Subang")
ax.plot(all_areas_df.index, all_areas_df["Kuantan"], linewidth=1.0, label="Kuantan")
ax.set_xlabel("Year", fontsize=5)
ax.set_ylabel("Rainfall (mm)", fontsize=5)
ax.tick_params(axis="x", labelsize=4)
ax.tick_params(axis="y", labelsize=4)
ax.legend(fontsize=4, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.caption("Time series plot of monthly rainfall from January 2014 to December 2023 for Alor Setar, Subang, and Kuantan.")
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LOAD SELECTED AREA DATA
# =========================
data = load_data(file, area)

# =========================
# TRAIN / TEST SPLIT
# =========================
if len(data) <= test_size + 12:
    st.error("Dataset is too short for the selected testing period.")
    st.stop()

train = data.iloc[:-test_size].copy()
test = data.iloc[-test_size:].copy()

# =========================
# BACKTEST - SELECTED MODEL
# =========================
if model_choice == "SARIMA":
    decomposition = None
    fitted_model, test_forecast, test_ci = fit_sarima(train["Rainfall"], test_size, area)
    future_model, future_forecast, future_ci = fit_sarima(data["Rainfall"], forecast_period, area)
else:
    decomposition, fitted_model, test_forecast, test_ci = fit_stl_arima(train["Rainfall"], test_size, area)
    _, future_model, future_forecast, future_ci = fit_stl_arima(data["Rainfall"], forecast_period, area)

# =========================
# BACKTEST - BOTH MODELS FOR COMPARISON
# =========================
_, sarima_test_forecast, _ = fit_sarima(train["Rainfall"], test_size, area)
_, _, stl_test_forecast, _ = fit_stl_arima(train["Rainfall"], test_size, area)

actual_test_series = test["Rainfall"].copy()

metrics_df, mae, rmse, mape = calculate_metrics(actual_test_series, test_forecast)
_, sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(actual_test_series, sarima_test_forecast)
_, stl_mae, stl_rmse, stl_mape = calculate_metrics(actual_test_series, stl_test_forecast)

best_model_name = get_best_model_name(sarima_mape, stl_mape)
risk_level, risk_class, risk_text = get_risk_level(future_forecast, data["Rainfall"])
insight_text = get_model_insight(area, model_choice, mape)

# =========================
# INSIGHT CARDS
# =========================
st.markdown(f"""
<div class="info-grid">
    <div class="info-card">
        <div class="info-card-title">📍 Selected Study Area</div>
        <div class="info-card-text">{area} is currently analysed using the <b>{model_choice}</b> forecasting framework.</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">🏆 Better Model on Test Set</div>
        <div class="info-card-text">Based on MAPE comparison, the better-performing model for this station is <b>{best_model_name}</b>.</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">🧠 Forecast Insight</div>
        <div class="info-card-text">{insight_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# CIRCULAR METRICS
# =========================
mae_text = f"{mae:.3f}" if not np.isnan(mae) else "NaN"
rmse_text = f"{rmse:.3f}" if not np.isnan(rmse) else "NaN"
mape_text = f"{mape:.2f}%" if not np.isnan(mape) else "NaN"

st.markdown(f"""
<div class="circle-metric-row">
    <div class="circle-metric">
        <div class="circle-metric-label">MAE</div>
        <div class="circle-metric-value">{mae_text}</div>
    </div>
    <div class="circle-metric">
        <div class="circle-metric-label">RMSE</div>
        <div class="circle-metric-value">{rmse_text}</div>
    </div>
    <div class="circle-metric">
        <div class="circle-metric-label">MAPE</div>
        <div class="circle-metric-value">{mape_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="risk-box {risk_class}">
    <div style="font-size:18px; font-weight:900; margin-bottom:6px;">Rainfall Risk Indicator: {risk_level}</div>
    <div style="font-size:16px; line-height:1.5;">{risk_text}</div>
</div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Observed Data", "Forecast Plot", "Forecast Table", "Decomposition", "Accuracy", "Model Comparison"]
)

with tab1:
    st.subheader(f"Observed Rainfall Data - {area}")
    st.dataframe(data.reset_index(), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(data.index, data["Rainfall"], linewidth=1.0, color="#1b9c85")
    ax.set_xlabel("Year", fontsize=5)
    ax.set_ylabel("Rainfall (mm)", fontsize=5)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption(f"Observed monthly rainfall from January 2014 to December 2023 for {area}.")

with tab2:
    st.subheader(f"{model_choice} Forecast with 95% CI - {area}")

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(data.index, data["Rainfall"], label="Observed", linewidth=1.0)
    ax.plot(future_forecast.index, future_forecast.values, label="Forecast", linewidth=1.0)
    ax.fill_between(
        future_ci.index,
        future_ci["Lower 95% CI"].values,
        future_ci["Upper 95% CI"].values,
        alpha=0.20,
        label="95% CI"
    )
    ax.axvline(data.index[-1], linestyle="--", linewidth=1)
    ax.set_xlabel("Year", fontsize=5)
    ax.set_ylabel("Rainfall (mm)", fontsize=5)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.legend(fontsize=4)
    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption(f"{model_choice} forecast plot with 95% confidence interval for {area}.")

with tab3:
    st.subheader("Forecast Table")

    forecast_df = pd.DataFrame({
        "Date": future_forecast.index,
        "Forecast": future_forecast.values,
        "Lower 95% CI": future_ci["Lower 95% CI"].values,
        "Upper 95% CI": future_ci["Upper 95% CI"].values
    })

    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"{area}_forecast.csv",
        mime="text/csv"
    )

with tab4:
    if model_choice == "STL-ARIMA":
        st.subheader("STL Decomposition")

        fig, axes = plt.subplots(4, 1, figsize=(7.0, 3.8), sharex=True)

        axes[0].plot(decomposition.observed, linewidth=0.8, color="#12355b")
        axes[0].set_title("Observed", fontsize=6)

        axes[1].plot(decomposition.trend, linewidth=0.8, color="#1b9c85")
        axes[1].set_title("Trend", fontsize=6)

        axes[2].plot(decomposition.seasonal, linewidth=0.8, color="#1565c0")
        axes[2].set_title("Seasonal", fontsize=6)

        axes[3].plot(decomposition.resid, linewidth=0.8, color="#e85d75")
        axes[3].set_title("Residual", fontsize=6)

        for ax_i in axes:
            ax_i.tick_params(axis="x", labelsize=4)
            ax_i.tick_params(axis="y", labelsize=4)
            ax_i.grid(True, linestyle="--", alpha=0.20)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption(f"STL decomposition of the rainfall series for {area}.")
    else:
        st.info("Decomposition is only shown for STL-ARIMA.")

with tab5:
    st.subheader(f"Forecast Accuracy on Test Set - {model_choice}")

    accuracy_df = metrics_df.reset_index().rename(columns={"index": "Date"})
    accuracy_df.columns = ["Date", "Actual", "Forecast"]
    st.dataframe(accuracy_df, use_container_width=True)

    accuracy_ci = test_ci.loc[metrics_df.index].copy()

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(train.index, train["Rainfall"], label="Train", linewidth=1.0)
    ax.plot(metrics_df.index, metrics_df["Actual"], label="Actual Test", linewidth=1.0)
    ax.plot(metrics_df.index, metrics_df["Predicted"], label="Forecast Test", linewidth=1.0)
    ax.fill_between(
        accuracy_ci.index,
        accuracy_ci["Lower 95% CI"].values,
        accuracy_ci["Upper 95% CI"].values,
        alpha=0.18,
        label="95% CI"
    )
    ax.axvline(train.index[-1], linestyle="--", linewidth=1)
    ax.set_xlabel("Year", fontsize=5)
    ax.set_ylabel("Rainfall (mm)", fontsize=5)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.legend(fontsize=4)
    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption(f"Observed and forecasted values on the testing dataset for {area} using {model_choice}.")

with tab6:
    st.subheader(f"Testing Dataset Comparison: Observed vs Forecasted ({area})")

    comparison_index = (
        actual_test_series.index
        .intersection(sarima_test_forecast.index)
        .intersection(stl_test_forecast.index)
    )

    comparison_df = pd.DataFrame({
        "Observed": actual_test_series.loc[comparison_index],
        "SARIMA Forecast": sarima_test_forecast.loc[comparison_index],
        "STL-ARIMA Forecast": stl_test_forecast.loc[comparison_index]
    }).dropna()

    st.dataframe(
        comparison_df.reset_index().rename(columns={"index": "Date"}),
        use_container_width=True
    )

    st.subheader("Observed vs Forecasted Values on Testing Dataset")

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(
        comparison_df.index,
        comparison_df["Observed"],
        linestyle="-",
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="Observed"
    )
    ax.plot(
        comparison_df.index,
        comparison_df["SARIMA Forecast"],
        linestyle="-.",
        marker="s",
        markersize=2.5,
        linewidth=1.0,
        label="SARIMA Forecast"
    )
    ax.plot(
        comparison_df.index,
        comparison_df["STL-ARIMA Forecast"],
        linestyle="--",
        marker="^",
        markersize=2.5,
        linewidth=1.0,
        label="STL-ARIMA Forecast"
    )

    ax.set_xlabel("Date", fontsize=5)
    ax.set_ylabel("Rainfall (mm)", fontsize=5)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.legend(fontsize=4)
    ax.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption(f"Observed and forecasted rainfall values on the testing dataset for {area}.")

    st.subheader("Model Performance Comparison")

    comparison_metrics = pd.DataFrame({
        "Model": ["SARIMA", "STL-ARIMA"],
        "MAE": [sarima_mae, stl_mae],
        "RMSE": [sarima_rmse, stl_rmse],
        "MAPE (%)": [sarima_mape, stl_mape]
    })

    st.dataframe(comparison_metrics, use_container_width=True)
