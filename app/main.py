import os

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

import streamlit as st


ROOT = os.path.dirname(os.path.dirname(__file__))


def resolve_path(path):
    return ROOT + path


def load_data(data_path):
    data = pd.read_csv(resolve_path(data_path))
    data["DATE"] = pd.to_datetime(data[["YEAR", "MONTH", "DAY"]])
    data.set_index("DATE", inplace=True)
    data = data[["CLOSE"]]
    return data


def train_arima_model(data, order):
    model = ARIMA(data["CLOSE"], order=order)
    model_fit = model.fit()
    return model_fit


def make_predictions(model, start, end):
    forecast = model.get_forecast(steps=end - start)
    return forecast.predicted_mean


st.set_page_config(
    page_title="PSX · Market Trend Predictor",
    page_icon="app/favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.title("ARIMA Model")
implementation = st.sidebar.selectbox("Select Implementation", ["Library", "Manual"])
forecast_period = st.sidebar.slider("Select Forecast Period", 1, 182, 31, 1)
forward_fill = st.sidebar.checkbox("Apply Forward Fill")

st.sidebar.subheader("Training Parameters")
p = st.sidebar.number_input("P", value=1, min_value=0)
d = st.sidebar.number_input("D", value=1, min_value=0)
q = st.sidebar.number_input("Q", value=31, min_value=0, max_value=365)

data_folder = "processed" if forward_fill else "raw"

training_set = load_data(f"/data/PSX/{data_folder}/train/data.csv")
validation_set = load_data(f"/data/PSX/{data_folder}/validate/data.csv")

st.title("PSX Market Trend Predictor")

st.subheader("Objective")
st.write(
    "The objective of this project is to predict the stock market trend for **Systems Limited**, a company listed on the Pakistan Stock Exchange (PSX). The prediction is based on historical stock price data scraped from the PSX website using a Selenium bot. The project utilizes both ARIMA (AutoRegressive Integrated Moving Average) and Linear Regression models (used by ARIMA) for prediction."
)

st.subheader("Data Collection")
st.write(
    """
    Data for Systems Limited was collected from the PSX website using a Selenium bot.

    - The dataset includes daily stock prices from 2020 to mid-2024.
    """
)

st.subheader("Data Processing")
st.write(
    """
    - The collected data was processed to handle missing values and prepare it for modeling.
    - Two versions of the dataset were prepared:
        1. **Forward-Filled Data:** Missing values were filled with the last observed value.
        2. **Data with Missing Values:** Missing values were retained as gaps in the data."""
)

st.subheader("Model Training and Validation")
st.write(
    """
    - **Training Set:** Data from 2020 to 2023 was used to train the models.
    - **Validation Set:** Data from January 2024 to June 2024 was reserved for validating the models.
    - **Models Used:**
        1. **ARIMA Model:** A time series forecasting model used to capture trends and seasonality in the data.
        2. **Linear Regression:** Utilized internally by ARIMA to capture linear relationships in AR and MA models.
"""
)

st.subheader("Trends")
st.write("**Systems Limited · Training Set (2020 - 2023)**")
st.line_chart(training_set, y_label="Price")
st.write("**Systems Limited · Validation Set (2024)**")
st.line_chart(
    validation_set[:forecast_period],
    color="#4CBB17",
    x_label="Days",
    y_label="Price",
)

if implementation == "Library":
    model_fit = train_arima_model(training_set, order=(1, 1, 1))
    library_predictions = make_predictions(model_fit, 0, forecast_period)
else:
    st.write("Placeholder for Manual ARIMA Implementation")

if implementation == "Manual":
    st.subheader("Comparative Analysis")
    st.write(
        "Placeholder for Comparative Analysis between Library and Manual Implementations"
    )
