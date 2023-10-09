import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to get historical stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1), data.index

# Function to prepare data for LSTM model
def prepare_data(data, sequence_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    sequences, labels = [], []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        labels.append(scaled_data[i + sequence_length])
    return np.array(sequences), np.array(labels), scaler

# Define LSTM model
def create_lstm_model(sequence_length):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function
def main():
    # Parameters
    ticker_symbol = "AAPL"  # Example: Apple Inc. stock
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    sequence_length = 10

    # Get stock data and prepare for LSTM model
    stock_data, dates = get_stock_data(ticker_symbol, start_date, end_date)
    sequences, labels, scaler = prepare_data(stock_data, sequence_length)

    # Split data into training and testing sets
    train_size = int(0.67 * len(stock_data))
    X_train, X_test, y_train, y_test = sequences[:train_size], sequences[train_size:], labels[:train_size], labels[train_size:]

    # Create and train LSTM model
    model = create_lstm_model(sequence_length)
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Make predictions
    predicted_stock_prices = model.predict(X_test)
    predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(dates[train_size + sequence_length:], predicted_stock_prices, label='Predicted Stock Prices', color='red')
    plt.plot(dates[train_size + sequence_length:], scaler.inverse_transform(y_test), label='Actual Stock Prices', color='blue')
    plt.title('Stock Price Prediction using LSTM')
    print("Predicted Stock Prices:")
    print(predicted_stock_prices)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show() 
