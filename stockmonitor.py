import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import requests

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data_yf(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Function to fetch stock data from Financial API
def fetch_stock_data_api(ticker, start_date, end_date, api_key="your_api_key"):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": api_key,
        "outputsize": "full"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(float)
        return df
    else:
        raise Exception("Error fetching data from API")

# Function to calculate moving average
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# AI-based suggestion using Linear Regression
def predict_trend(data):
    data = data.reset_index()
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    trend = "Upward" if model.coef_[0] > 0 else "Downward"
    return trend

# Streamlit App
st.title("Stock Dashboard")

# Sidebar for user inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
start_date = st.sidebar.date_input("Start Date:", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date:", datetime.date.today())
window_size = st.sidebar.slider("Moving Average Window (days):", min_value=5, max_value=50, value=20)
data_source = st.sidebar.selectbox("Select Data Source:", ["Yahoo Finance", "Financial API"])
api_key = st.sidebar.text_input("Enter Financial API Key (if using API):", "", type="password")

# Fetch and display stock data
if st.sidebar.button("Fetch Data"):
    try:
        if data_source == "Yahoo Finance":
            data = fetch_stock_data_yf(ticker, start_date, end_date)
        elif data_source == "Financial API":
            if not api_key:
                st.error("API Key is required for Financial API.")
                raise ValueError("Missing API Key")
            data = fetch_stock_data_api(ticker, start_date, end_date, api_key)

        # Display data
        st.subheader(f"Stock Data for {ticker} from {start_date} to {end_date}")
        st.dataframe(data.tail())

        # Plot closing prices
        st.subheader("Closing Prices")
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label="Closing Price")
        ma = moving_average(data, window_size)
        plt.plot(ma, label=f"{window_size}-Day Moving Average", linestyle="--")
        plt.legend()
        plt.title(f"{ticker} Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(plt)

        # Trend prediction
        trend = predict_trend(data)
        st.subheader("AI Suggestion")
        st.write(f"The stock shows a {trend} trend based on historical data.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
