# ============================================================
# AGENTIC AI ELECTRICITY DEMAND FORECASTING DASHBOARD
# ============================================================
import os
from dotenv import load_dotenv
from google import genai
from pathlib import Path
import json

import holidays
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import tensorflow as tf

load_dotenv()
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside dashboard/
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
PLOTS_DIR = PROJECT_DIR / "plots"

DATA_PATH = DATA_DIR / "tetuan_features.parquet"

MODEL_PATH = MODEL_DIR / "cnn_lstm_attention.keras"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

METRICS_PATH = REPORT_DIR / "metrics.json"
SHAP_PATH = REPORT_DIR / "shap_global_importance.csv"
NEXT_DAY_PATH = REPORT_DIR / "next_24_hour_forecast.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .agent-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 14px;

        /* IMPORTANT FIX */
        background-color: #1f2937;
        color: #f9fafb;

        font-size: 16px;
        line-height: 1.6;
    }

    .agent-card b {
        color: #ffffff;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE VALIDATION
# ============================================================

required_files = {
    "Feature dataset": DATA_PATH,
    "Forecast model": MODEL_PATH,
    "Feature scaler": FEATURE_SCALER_PATH,
    "Target scaler": TARGET_SCALER_PATH,
    "Metadata": METADATA_PATH
}

missing_files = [
    f"{name}: {path}"
    for name, path in required_files.items()
    if not path.exists()
]

if missing_files:

    st.error("Some required project files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model_and_scalers():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    feature_scaler = joblib.load(
        FEATURE_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    return model, feature_scaler, target_scaler


@st.cache_data
def load_project_data():

    df = pd.read_parquet(
        DATA_PATH
    )

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        metadata = json.load(file)

    metrics = {}

    if METRICS_PATH.exists():

        with open(
            METRICS_PATH,
            "r"
        ) as file:

            metrics = json.load(file)

    return df, metadata, metrics


forecast_model, feature_scaler, target_scaler = (
    load_model_and_scalers()
)

df, metadata, metrics = (
    load_project_data()
)

FEATURES = metadata["features"]
TARGET = metadata["target"]

SEQUENCE_LENGTH = int(
    metadata["sequence_length"]
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(
    _model,
    _df,
    _feature_scaler
):

    background_count = min(
        40,
        max(
            1,
            len(_df) - SEQUENCE_LENGTH
        )
    )

    positions = np.linspace(
        SEQUENCE_LENGTH,
        len(_df) - 1,
        background_count,
        dtype=int
    )

    background_sequences = []

    for end_index in positions:

        sequence = (
            _df[FEATURES]
            .iloc[
                end_index - SEQUENCE_LENGTH:
                end_index
            ]
        )

        scaled = (
            _feature_scaler
            .transform(sequence)
            .astype(np.float32)
        )

        background_sequences.append(
            scaled
        )

    background = np.array(
        background_sequences,
        dtype=np.float32
    )

    explainer = shap.GradientExplainer(
        _model,
        background
    )

    return explainer


shap_explainer = create_shap_explainer(
    forecast_model,
    df,
    feature_scaler
)

@st.cache_resource
def get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    try:
        client = genai.Client(
            api_key=api_key
        )

        return client

    except Exception:
        return None


gemini_client = get_gemini_client()
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_risk(prediction):

    target_values = df[TARGET]

    moderate = float(
        target_values.quantile(0.60)
    )

    high = float(
        target_values.quantile(0.80)
    )

    critical = float(
        target_values.quantile(0.95)
    )

    if prediction >= critical:
        return "CRITICAL"

    elif prediction >= high:
        return "HIGH"

    elif prediction >= moderate:
        return "MODERATE"

    else:
        return "NORMAL"


def predict_manual(
    temperature,
    humidity,
    wind_speed,
    general_diffuse,
    diffuse
):

    latest_sequence = (
        df[FEATURES]
        .iloc[-SEQUENCE_LENGTH:]
        .copy()
    )

    latest_sequence.loc[
        latest_sequence.index[-1],
        "Temperature"
    ] = temperature

    latest_sequence.loc[
        latest_sequence.index[-1],
        "Humidity"
    ] = humidity

    latest_sequence.loc[
        latest_sequence.index[-1],
        "WindSpeed"
    ] = wind_speed

    latest_sequence.loc[
        latest_sequence.index[-1],
        "GeneralDiffuseFlows"
    ] = general_diffuse

    latest_sequence.loc[
        latest_sequence.index[-1],
        "DiffuseFlows"
    ] = diffuse

    scaled_sequence = (
        feature_scaler
        .transform(
            latest_sequence[FEATURES]
        )
        .astype(np.float32)
    )

    model_input = np.expand_dims(
        scaled_sequence,
        axis=0
    )

    prediction_scaled = (
        forecast_model.predict(
            model_input,
            verbose=0
        )
    )

    prediction = float(
        target_scaler
        .inverse_transform(
            prediction_scaled
        )[0, 0]
    )

    return prediction, model_input


def get_local_shap(model_input):

    try:

        raw_values = shap_explainer.shap_values(
            model_input
        )

        values = (
            raw_values[0]
            if isinstance(raw_values, list)
            else raw_values
        )

        values = np.asarray(values)

        if (
            values.ndim == 4
            and values.shape[-1] == 1
        ):
            values = values[..., 0]

        if values.ndim != 3:
            return {}

        feature_values = (
            values[0]
            .sum(axis=0)
        )

        return {
            feature: float(value)
            for feature, value
            in zip(
                FEATURES,
                feature_values
            )
        }

    except Exception as error:

        st.warning(
            f"Local SHAP unavailable: {error}"
        )

        return {}


def get_top_factors(
    contributions,
    n=5
):

    return sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:n]


def analyze_weather(
    temperature,
    contributions
):

    weather_features = {
        "Temperature",
        "Humidity",
        "WindSpeed",
        "GeneralDiffuseFlows",
        "DiffuseFlows"
    }

    weather_values = {
        name: value
        for name, value
        in contributions.items()
        if name in weather_features
    }

    dominant = sorted(
        weather_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:3]

    if temperature > 30:
        status = "Hot"

    elif temperature < 15:
        status = "Cold"

    else:
        status = "Moderate"

    reasons = []

    for name, value in dominant:

        direction = (
            "increased"
            if value > 0
            else "reduced"
        )

        reasons.append(
            f"{name} {direction} demand "
            f"({value:+.4f})"
        )

    return {
        "status": status,
        "reasons": reasons
    }


def analyze_calendar(
    timestamp,
    contributions
):

    weekend = int(
        timestamp.dayofweek >= 5
    )

    morocco_calendar = (
        holidays.country_holidays(
            "MA",
            years=[
                timestamp.year
            ]
        )
    )

    holiday = int(
        timestamp.date()
        in morocco_calendar
    )

    if holiday:
        status = "Holiday"

    elif weekend:
        status = "Weekend"

    else:
        status = "Working Day"

    calendar_features = {
        "Weekend",
        "Holiday",
        "HourSin",
        "HourCos",
        "DayOfWeekSin",
        "DayOfWeekCos",
        "MonthSin",
        "MonthCos"
    }

    values = {
        name: value
        for name, value
        in contributions.items()
        if name in calendar_features
    }

    dominant = sorted(
        values.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:3]

    reasons = []

    for name, value in dominant:

        direction = (
            "increased"
            if value > 0
            else "reduced"
        )

        reasons.append(
            f"{name} {direction} demand "
            f"({value:+.4f})"
        )

    return {
        "status": status,
        "weekend": weekend,
        "holiday": holiday,
        "reasons": reasons
    }


def generate_explanation(
    prediction,
    risk,
    weather,
    calendar,
    factors
):

    factor_text = ", ".join(
        [
            f"{name}: {value:+.4f}"
            for name, value
            in factors
        ]
    )

    if not factor_text:
        factor_text = (
            "SHAP factors unavailable"
        )

    # --------------------------------------------------------
    # Gemini LLM Explanation
    # --------------------------------------------------------

    if gemini_client is not None:

        prompt = f"""
You are an electricity-demand explanation agent.

Explain the forecasting result in simple,
clear and factual language.

Do not invent any information.

Forecast Details:
Predicted electricity demand: {prediction:.2f}
Risk level: {risk}
Weather status: {weather['status']}
Calendar status: {calendar['status']}
Top SHAP factors: {factor_text}

Explain:
1. What the predicted demand means.
2. Whether the demand is risky.
3. How weather influenced the forecast.
4. How calendar conditions influenced the forecast.
5. Which features influenced the prediction most.

Write only 4 to 6 concise sentences.
"""

        try:

            response = (
                gemini_client.models
                .generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            )

            if (
                response
                and response.text
            ):

                return (
                    response.text.strip(),
                    "Gemini LLM"
                )

        except Exception as error:

            print(
                "Gemini failed:",
                error
            )

    # --------------------------------------------------------
    # Rule-based fallback
    # --------------------------------------------------------

    fallback = (
        f"The CNN-LSTM-Attention model predicts "
        f"an electricity demand of "
        f"{prediction:,.2f} units. "
        f"The demand risk is classified as "
        f"{risk}. "
        f"The Weather Agent identifies the "
        f"weather condition as "
        f"{weather['status']}, while the "
        f"Calendar Agent identifies the date as "
        f"{calendar['status']}. "
        f"The most influential features are "
        f"{factor_text}."
    )

    return (
        fallback,
        "Rule-Based Fallback"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚡ Agentic AI"
)

st.sidebar.caption(
    "Electricity Demand Forecasting"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Manual Forecast",
        "Next-Day Forecast",
        "Explainability",
        "Agent Insights"
    ]
)

st.sidebar.divider()

st.sidebar.success(
    "CNN-LSTM-Attention"
)

st.sidebar.info(
    f"Sequence Window: "
    f"{SEQUENCE_LENGTH} records"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    ⚡ Agentic AI Electricity Demand Forecasting
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Explainable CNN-LSTM-Attention forecasting
    with weather, calendar, SHAP and risk intelligence
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader(
        "System Overview"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Model",
        "CNN-LSTM-Attention"
    )

    c2.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

    c3.metric(
        "R² Score",
        f"{metrics.get('R2', 0):.4f}"
    )

    c4.metric(
        "RMSE",
        f"{metrics.get('RMSE', 0):,.2f}"
    )

    st.divider()

    st.subheader(
        "Historical Electricity Demand"
    )

    history_df = (
        df[[TARGET]]
        .tail(1000)
        .reset_index()
    )

    time_column = (
        history_df.columns[0]
    )

    fig = px.line(
        history_df,
        x=time_column,
        y=TARGET,
        title=(
            "Recent Zone 1 Electricity Demand"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Average Demand",
        f"{df[TARGET].mean():,.2f}"
    )

    c2.metric(
        "Maximum Demand",
        f"{df[TARGET].max():,.2f}"
    )

    c3.metric(
        "Minimum Demand",
        f"{df[TARGET].min():,.2f}"
    )

    st.divider()

    st.subheader(
        "Weather-Demand Correlation"
    )

    correlation_features = [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "GeneralDiffuseFlows",
        "DiffuseFlows",
        TARGET
    ]

    available_features = [
        column
        for column in correlation_features
        if column in df.columns
    ]

    corr = (
        df[available_features]
        .corr()
    )

    heatmap = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — MANUAL FORECAST
# ============================================================

elif page == "Manual Forecast":

    st.subheader(
        "Manual Electricity Demand Forecast"
    )

    st.info(
        "Enter the weather conditions and forecast date. "
        "The system combines them with the latest "
        "24-hour electricity consumption sequence."
    )

    left, right = (
        st.columns(2)
    )

    with left:

        temperature = (
            st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=20.0,
                step=0.1
            )
        )

        humidity = (
            st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=0.1
            )
        )

        wind_speed = (
            st.number_input(
                "Wind Speed",
                min_value=0.0,
                value=0.1,
                step=0.01
            )
        )

    with right:

        general_diffuse = (
            st.number_input(
                "General Diffuse Flows",
                min_value=0.0,
                value=100.0,
                step=1.0
            )
        )

        diffuse = (
            st.number_input(
                "Diffuse Flows",
                min_value=0.0,
                value=50.0,
                step=1.0
            )
        )

        recommended_datetime = (
            df.index[-1]
            + pd.Timedelta(minutes=10)
        )

        forecast_date = (
            st.date_input(
                "Forecast Date",
                value=recommended_datetime.date()
            )
        )

        forecast_time = (
            st.time_input(
                "Forecast Time",
                value=recommended_datetime.time()
            )
        )

        selected_datetime = (
            pd.Timestamp.combine(
                forecast_date,
                forecast_time
            )
        )

    predict_clicked = (
        st.button(
            "⚡ Predict Demand",
            use_container_width=True
        )
    )

    if predict_clicked:

        with st.spinner(
            "Running Agentic AI workflow..."
        ):

            # Forecast Agent
            prediction, model_input = (
                predict_manual(
                    temperature,
                    humidity,
                    wind_speed,
                    general_diffuse,
                    diffuse
                )
            )

            # SHAP Agent
            contributions = (
                get_local_shap(
                    model_input
                )
            )

            factors = (
                get_top_factors(
                    contributions,
                    n=5
                )
            )

            # Weather Agent
            weather = (
                analyze_weather(
                    temperature,
                    contributions
                )
            )

            # Calendar Agent
            calendar = (
                analyze_calendar(
                    selected_datetime,
                    contributions
                )
            )

            # Risk Agent
            risk = (
                classify_risk(
                    prediction
                )
            )

            # Explanation Agent
            explanation, explanation_source = (
                generate_explanation(
                    prediction,
                    risk,
                    weather,
                    calendar,
                    factors
                )
            )

        st.success(
            "Agentic AI forecast completed."
        )

        st.subheader(
            "Forecast Result"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "⚡ Predicted Demand",
            f"{prediction:,.2f}"
        )

        c2.metric(
            "⚠️ Risk Level",
            risk
        )

        c3.metric(
            "🌤 Weather",
            weather["status"]
        )

        c4.metric(
            "📅 Calendar",
            calendar["status"]
        )

        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        st.subheader(
            "Demand Risk Visualization"
        )

        minimum = float(
            df[TARGET].min()
        )

        maximum = float(
            df[TARGET].max()
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={
                    "text":
                    "Predicted Electricity Demand"
                },
                gauge={
                    "axis": {
                        "range": [
                            minimum,
                            maximum
                        ]
                    },
                    "threshold": {
                        "line": {
                            "width": 5
                        },
                        "value": prediction
                    }
                }
            )
        )

        gauge.update_layout(
            height=350
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.subheader(
            "🔍 Prediction Explainability"
        )

        if factors:

            shap_display = (
                pd.DataFrame(
                    factors,
                    columns=[
                        "Feature",
                        "SHAP Contribution"
                    ]
                )
            )

            shap_display[
                "Absolute Contribution"
            ] = (
                shap_display[
                    "SHAP Contribution"
                ].abs()
            )

            shap_display = (
                shap_display.sort_values(
                    "Absolute Contribution",
                    ascending=True
                )
            )

            shap_chart = px.bar(
                shap_display,
                x="SHAP Contribution",
                y="Feature",
                orientation="h",
                title=(
                    "Top 5 Factors Affecting "
                    "This Forecast"
                )
            )

            st.plotly_chart(
                shap_chart,
                use_container_width=True
            )

            st.write(
                "### Feature Interpretation"
            )

            for name, value in factors:

                if value > 0:

                    st.write(
                        f"⬆️ **{name}** increased "
                        f"the forecast ({value:+.4f})"
                    )

                else:

                    st.write(
                        f"⬇️ **{name}** reduced "
                        f"the forecast ({value:+.4f})"
                    )

        else:

            st.warning(
                "Local SHAP explanation could "
                "not be generated."
            )

        # ----------------------------------------------------
        # AGENT ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "🤖 Agentic AI Analysis"
        )

        agent1, agent2, agent3 = (
            st.columns(3)
        )

        with agent1:

            st.markdown(
                "### 🌤 Weather Agent"
            )

            st.write(
                "**Status:**",
                weather["status"]
            )

            if weather["reasons"]:

                for reason in (
                    weather["reasons"]
                ):

                    st.write(
                        "•",
                        reason
                    )

            else:

                st.write(
                    "No strong weather "
                    "contribution identified."
                )

        with agent2:

            st.markdown(
                "### 📅 Calendar Agent"
            )

            st.write(
                "**Status:**",
                calendar["status"]
            )

            st.write(
                "**Weekend:**",
                "Yes"
                if calendar["weekend"]
                else "No"
            )

            st.write(
                "**Holiday:**",
                "Yes"
                if calendar["holiday"]
                else "No"
            )

            if calendar["reasons"]:

                for reason in (
                    calendar["reasons"]
                ):

                    st.write(
                        "•",
                        reason
                    )

        with agent3:

            st.markdown(
                "### ⚠️ Risk Agent"
            )

            if risk == "CRITICAL":

                st.error(
                    "CRITICAL DEMAND"
                )

            elif risk == "HIGH":

                st.error(
                    "HIGH DEMAND"
                )

            elif risk == "MODERATE":

                st.warning(
                    "MODERATE DEMAND"
                )

            else:

                st.success(
                    "NORMAL DEMAND"
                )

        # ----------------------------------------------------
        # EXPLANATION AGENT
        # ----------------------------------------------------

        st.subheader(
            "💬 AI Explanation"
        )

        st.info(
            explanation
        )
        st.caption(
            f"Explanation Source: {explanation_source}")

        # ----------------------------------------------------
        # WORKFLOW STATUS
        # ----------------------------------------------------

        st.subheader(
            "🔄 Multi-Agent Workflow Status"
        )

        workflow = pd.DataFrame(
            {
                "Agent": [
                    "Data Agent",
                    "Forecast Agent",
                    "SHAP Agent",
                    "Weather Agent",
                    "Calendar Agent",
                    "Risk Agent",
                    "Explanation Agent"
                ],

                "Status": [
                    "Completed",
                    "Completed",
                    (
                        "Completed"
                        if factors
                        else "Unavailable"
                    ),
                    "Completed",
                    "Completed",
                    "Completed",
                    "Completed"
                ]
            }
        )

        st.dataframe(
            workflow,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 3 — NEXT-DAY FORECAST
# ============================================================

elif page == "Next-Day Forecast":

    st.subheader(
        "Next 24-Hour Electricity Demand Forecast"
    )

    if not NEXT_DAY_PATH.exists():

        st.warning(
            "next_24_hour_forecast.csv was not found "
            "inside reports/. Run the next-day forecasting "
            "cells in Colab first."
        )

    else:

        next_day = (
            pd.read_csv(
                NEXT_DAY_PATH
            )
        )

        if "Datetime" in next_day.columns:

            next_day[
                "Datetime"
            ] = pd.to_datetime(
                next_day[
                    "Datetime"
                ]
            )

        next_day_fig = (
            px.line(
                next_day,
                x="Datetime",
                y="PredictedDemand",
                markers=True,
                title=(
                    "Next-Day Electricity Demand Profile"
                )
            )
        )

        st.plotly_chart(
            next_day_fig,
            use_container_width=True
        )

        average_demand = float(
            next_day[
                "PredictedDemand"
            ].mean()
        )

        maximum_demand = float(
            next_day[
                "PredictedDemand"
            ].max()
        )

        minimum_demand = float(
            next_day[
                "PredictedDemand"
            ].min()
        )

        peak_index = (
            next_day[
                "PredictedDemand"
            ].idxmax()
        )

        peak_time = (
            next_day.loc[
                peak_index,
                "Datetime"
            ]
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Average Demand",
            f"{average_demand:,.2f}"
        )

        c2.metric(
            "Maximum Demand",
            f"{maximum_demand:,.2f}"
        )

        c3.metric(
            "Minimum Demand",
            f"{minimum_demand:,.2f}"
        )

        c4.metric(
            "Peak Time",
            peak_time.strftime(
                "%H:%M"
            )
        )

        if "RiskLevel" in (
            next_day.columns
        ):

            st.subheader(
                "Risk Distribution"
            )

            risk_counts = (
                next_day[
                    "RiskLevel"
                ]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = [
                "RiskLevel",
                "Count"
            ]

            risk_fig = px.bar(
                risk_counts,
                x="RiskLevel",
                y="Count",
                title=(
                    "Next-Day Demand Risk Distribution"
                )
            )

            st.plotly_chart(
                risk_fig,
                use_container_width=True
            )

        st.subheader(
            "Forecast Records"
        )

        st.dataframe(
            next_day,
            use_container_width=True,
            height=420
        )


# ============================================================
# PAGE 4 — EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.subheader(
        "SHAP Explainability"
    )

    st.write(
        "This page displays the global importance "
        "of features influencing electricity demand."
    )

    if not SHAP_PATH.exists():

        st.warning(
            "shap_global_importance.csv was not found "
            "inside reports/."
        )

    else:

        shap_df = (
            pd.read_csv(
                SHAP_PATH
            )
        )

        shap_df = (
            shap_df.sort_values(
                "MeanAbsoluteSHAP",
                ascending=False
            )
            .head(15)
        )

        plot_df = (
            shap_df.sort_values(
                "MeanAbsoluteSHAP",
                ascending=True
            )
        )

        shap_fig = px.bar(
            plot_df,
            x="MeanAbsoluteSHAP",
            y="Feature",
            orientation="h",
            title=(
                "Top 15 Global SHAP Feature Importance"
            )
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True
        )

        strongest = (
            shap_df.iloc[0]
        )

        st.success(
            "Most influential feature: "
            f"{strongest['Feature']}"
        )

        st.dataframe(
            shap_df,
            use_container_width=True
        )


# ============================================================
# PAGE 5 — AGENT INSIGHTS
# ============================================================

elif page == "Agent Insights":

    st.subheader(
        "Agentic AI Decision Pipeline"
    )

    st.write(
        "The forecasting framework uses multiple "
        "specialized agents for prediction, explainability, "
        "context analysis and decision support."
    )

    st.markdown(
        """
        <div class="agent-card">
        <b>1. Data Agent</b><br>
        Collects electricity demand, weather and calendar information.
        </div>

        <div class="agent-card">
        <b>2. Forecast Agent</b><br>
        Uses the CNN-LSTM-Attention model to predict electricity demand.
        </div>

        <div class="agent-card">
        <b>3. SHAP Agent</b><br>
        Identifies the most influential features affecting the forecast.
        </div>

        <div class="agent-card">
        <b>4. Weather Agent</b><br>
        Analyzes weather-related effects on electricity demand.
        </div>

        <div class="agent-card">
        <b>5. Calendar Agent</b><br>
        Evaluates weekend, holiday and working-day conditions.
        </div>

        <div class="agent-card">
        <b>6. Risk Agent</b><br>
        Classifies predicted demand as Normal, Moderate, High or Critical.
        </div>

        <div class="agent-card">
        <b>7. Explanation Agent</b><br>
        Produces a human-readable interpretation of the forecast.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Multi-Agent Workflow"
    )

    st.code(
        """
START
  ↓
Data Agent
  ↓
Forecast Agent
  ↓
SHAP Agent
  ↓
Weather Agent
  ↓
Calendar Agent
  ↓
Risk Agent
  ↓
Explanation Agent
  ↓
END
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic AI Framework for Explainable "
    "Electricity Demand Forecasting"
)