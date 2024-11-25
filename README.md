---

# Stock Price Prediction and Trading Strategy System

This project provides a stock price prediction and trading recommendation tool. Using a hybrid model combining Prophet (for trend and seasonality) and LSTM (for sequential learning) to predict future stock prices this tool includes an interactive user interface, allowing users to enter stock tickers, view predictions, and receive buy/sell/hold recommendations. The goal of this project is to forecast stock prices based on historical data and provide actionable trading strategies (buy, sell, hold) using an intuitive Streamlit interface. This software includes modules for data collection, preprocessing, feature engineering, modeling, and visualization.

---

## Features

- **Stock Price Forecasting**: Uses a hybrid Prophet and bidirectional LSTM model for accurate stock price predictions.
- **Technical Indicators**: Incorporates SMA, EMA, Bollinger Bands, and other indicators for enriched features.
- **Trading Strategy Recommendations**: Suggests buy, sell, or hold actions based on predictions.
- **Interactive Interface**: Streamlit-powered UI for entering stock tickers, displaying predictions, and viewing trading recommendations.

## Technology Stack

- **Frontend**: Streamlit for user interface.
- **Backend**: Flask for pipeline and API management.
- **Machine Learning**: PyTorch for LSTM modeling and Facebook Prophet for time-series analysis.
- **Data Processing**: Pandas and NumPy for data manipulation.
- **Testing**: Pytest for unit and integration tests, Locust for performance testing, OWASP ZAP for security testing.
- **CI/CD**: GitHub Actions for automated testing.
