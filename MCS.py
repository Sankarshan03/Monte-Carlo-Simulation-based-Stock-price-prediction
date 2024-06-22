import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ttest_ind
import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense #type:ignore
import yfinance as yf

# Define the list of stocks
tickers = ['NVDA', 'SIRI', 'AAPL', 'BAC', 'CSCO']

# Download historical stock price data
data = yf.download(tickers=tickers, start='2010-01-01', end='2023-01-01')['Adj Close']

# Calculate log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Monte Carlo Simulation function
def monte_carlo_simulation(S0, mu, sigma, T, M, I):
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        Z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S

# Plotting the results of Monte Carlo Simulation for each stock
for ticker in tickers:
    S0 = data[ticker][-1]
    mu = log_returns[ticker].mean()
    sigma = log_returns[ticker].std()
    T = 1.0
    M = 252
    I = 10000
    
    simulations = monte_carlo_simulation(S0, mu, sigma, T, M, I)
    
    plt.figure(figsize=(10, 6))
    plt.plot(simulations[:, :10])
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title(f'Monte Carlo Simulations of {ticker} Stock Prices')
    plt.show()

# ANN for Stock Price Prediction
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Training and testing the ANN model for each stock
for ticker in tickers:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[ticker].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(Dense(50, input_shape=(time_step, 1), activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train = scaler.inverse_transform([Y_train])
    Y_test = scaler.inverse_transform([Y_test])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data[ticker], label='Actual Stock Price')
    plt.plot(range(time_step, time_step + len(train_predict)), train_predict, label='Train Predict')
    plt.plot(range(time_step + len(train_predict) + 1, time_step + len(train_predict) + 1 + len(test_predict)), test_predict, label='Test Predict')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Prediction of {ticker} using ANN')
    plt.legend()
    plt.show()

    # Statistical Analysis
    actual_prices = data[ticker][time_step + len(train_predict) + 1:].values
    predicted_prices = test_predict.flatten()

    correlation, _ = pearsonr(actual_prices, predicted_prices)
    print(f'Pearson Correlation Coefficient for {ticker}: {correlation:.4f}')

    t_stat, p_value = ttest_ind(actual_prices, predicted_prices)
    print(f'T-Test Statistic for {ticker}: {t_stat:.4f}, P-Value: {p_value:.4f}')

    mse = mean_squared_error(actual_prices, predicted_prices)
    print(f'Mean Squared Error (MSE) of ANN Predictions for {ticker}: {mse:.4f}')

    # Plot correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_prices, predicted_prices, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs. Predicted Stock Prices for {ticker}')
    plt.show()
