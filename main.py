import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("googl_daily_prices.csv")
df.drop(columns="date", inplace=True)
features = df[["1. open", "3. low", "2. high", "5. volume"]] # do not use values so that ct will match the columns correctly
labels = df["4. close"]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer([
        ("normalize", StandardScaler(), ["1. open", "2. high", "3. low", "5. volume"])
], remainder='passthrough')
features_train_scaled = preprocessor.fit_transform(features_train)
features_test_scaled = preprocessor.transform(features_test)
stop = EarlyStopping(monitor="val_loss", patience=40, mode="min", verbose=1, restore_best_weights=True)
def build_model():
    model = Sequential([
        layers.Dense(4, activation="relu"),
        layers.Dense(4, activation='relu'),
        layers.Dense(1)
    ])
    opt = Adam(learning_rate=0.01)
    model.compile(
        optimizer = opt,
        loss = "mean_squared_error",
        metrics = ["mae","mse"]
    )
    return model

google_tracker = build_model()

history = google_tracker.fit(features_train_scaled, labels_train, epochs=50, callbacks=[stop], validation_split=0.2)
google_tracker.save("google_tracker.keras")
loss, mae, mse =  google_tracker.evaluate(features_test_scaled, labels_test)

plt.plot(history.history["val_mae"], color='chartreuse', ls='--', label="Validation MAE")
plt.plot(history.history["val_mse"], color='olivedrab', ls='--', label="Validation MSE")
plt.plot(history.history["mse"], color='saddlebrown', ls='--', label="Training MSE")
plt.plot(history.history["mae"], color='slateblue', ls='--', label="Training MAE")
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()

print(f"MAE: {mae: .2f}, MSE: {mse: .2f}")
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features_train_scaled, labels_train)
y_pred = dummy_regr.predict(features_test_scaled)
mae_baseline = mean_absolute_error(labels_test, y_pred)
print("Baseline MAE:", mae_baseline)

user_input = pd.DataFrame([[178.5, 179.67, 179.53, 21689729]], columns=["1. open", "2. high", "3. low", "5. volume"])
scaled_input = preprocessor.transform(user_input)
prediction = google_tracker.predict(scaled_input)

print(f"Predicted closing price is: {prediction[0][0]: .2f}")
