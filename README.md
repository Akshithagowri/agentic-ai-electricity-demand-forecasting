# Agentic AI Framework for Explainable Electricity Demand Forecasting

This project presents an Agentic AI-based electricity demand forecasting framework that combines deep learning, explainable AI, and multi-agent reasoning to generate accurate and interpretable electricity demand predictions.

The system uses historical electricity consumption, weather conditions, calendar information, lag features, and rolling statistics to forecast future demand. A hybrid CNN–LSTM–Attention model is used as the core forecasting model, while SHAP is used to explain the contribution of individual features.

## Main Features

* Electricity demand forecasting using CNN–LSTM–Attention
* Weather-based demand analysis
* Calendar, weekend, and holiday analysis
* SHAP-based explainability
* Risk classification
* Multi-agent reasoning
* Human-readable explanation generation
* Manual demand forecasting
* Next-day 24-hour recursive forecasting
* Interactive Streamlit dashboard

## Dataset

The project uses the Tetuan City Power Consumption dataset.

Main features include:

* Temperature
* Humidity
* Wind Speed
* General Diffuse Flows
* Diffuse Flows
* Zone 1 Electricity Consumption
* Temporal features
* Weekend and holiday features
* Lag features
* Rolling statistics

Zone 1 electricity consumption is used as the forecasting target.

## Model Architecture

The forecasting model uses:

CNN → LSTM → Attention → Dense Layers → Electricity Demand Prediction

CNN extracts local patterns from the input time series, LSTM learns temporal dependencies, and the Attention mechanism focuses on the most relevant time steps.

## Agentic AI Components

The system consists of multiple specialized agents:

* Data Agent
* Forecast Agent
* SHAP Explainability Agent
* Weather Agent
* Calendar Agent
* Risk Agent
* Explanation Agent

The agents collaboratively analyze the prediction and provide contextual information for decision support.

## Explainable AI

SHAP is used to identify the contribution of each input feature toward the predicted electricity demand.

The dashboard displays:

* Global SHAP feature importance
* Local SHAP feature contributions
* Features that increase demand
* Features that reduce demand

## Next-Day Forecasting

The model initially performs one-step forecasting at ten-minute intervals.

To generate a full next-day demand profile, recursive forecasting is used.

Since the dataset contains six records per hour:

24 × 6 = 144 predictions

The system generates 144 sequential predictions to represent the next 24 hours.

## Dashboard

The project includes a Streamlit dashboard with the following pages:

* Overview
* Manual Forecast
* Next-Day Forecast
* Explainability
* Agent Insights

The dashboard provides interactive visualizations for electricity demand, SHAP values, risk levels, and agent outputs.

## Project Structure

```text
electricity_forecasting_project/
│
├── dashboard/
│   └── app.py
│
├── data/
│
├── models/
│
├── reports/
│
├── plots/
│
├── config/
│
├── docs/
│
├── simulator/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Open the project directory:

```bash
cd electricity_forecasting_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

## Technologies Used

* Python
* TensorFlow / Keras
* CNN
* LSTM
* Attention Mechanism
* SHAP
* LangGraph
* Streamlit
* Plotly
* Pandas
* NumPy
* Scikit-learn
* Gemini LLM

## Future Improvements

Future versions can integrate real-time electricity demand data, weather APIs, more recent datasets, direct multi-step forecasting, and cloud deployment for real-time energy-management applications.
