# Google Closing Price Predictor (Neural Network)

A deep learning model that predicts **Google’s stock closing price** based on historical daily trading data. The model uses a fully connected neural network trained on open, high, low, and volume data to estimate the close price — with impressive accuracy.


## Model Summary

This neural network learns the underlying patterns in Google's stock data using:

- Dense neural network architecture (3 layers)
- Normalized input features using `StandardScaler`
- Evaluation vs. a naive baseline model (`DummyRegressor`)
- Prediction visualization using `matplotlib`



## Model Accuracy

**Example Evaluation Output:**

- **MAE (Model):** 5.55  
- **MSE (Model):** 94.90  
- **Baseline MAE:** 452.05  

**Example Prediction:**

- **Input:** `[178.5, 179.67, 179.53, 21689729]`
- **Actual Close Price:** `179.53`  
- **Predicted Close Price:** `179.21`


Features Used

| Feature        | Description              |
|----------------|--------------------------|
| `1. open`      | Opening price            |
| `2. high`      | Daily high               |
| `3. low`       | Daily low                |
| `5. volume`    | Trading volume           |



## How It Works

### 1. Preprocessing

- CSV file (`googl_daily_prices.csv`) is read
- Date column is dropped
- Features normalized using `StandardScaler` in a `ColumnTransformer`

### 2. Neural Network Model

```python
model = Sequential([
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

