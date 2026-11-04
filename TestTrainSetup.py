# ============================================================
# SIGMOID SISTERS ASF
# BZAN 554 - Group Assignment 1
# Neural Network Development Model
# ============================================================

# ============================================================
# IMPORT PACKAGES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Reproducibility
np.random.seed(1)
tf.random.set_seed(1)


# ============================================================
# 1. LOAD AND SET UP DATA
# ============================================================

data = pd.read_csv("pricing.csv")

# Keep only assignment variables
data = data[
    ["sku", "price", "order", "duration", "category", "quantity"]
].copy()

# Predictors
X = data[
    ["sku", "price", "order", "duration", "category"]
].to_numpy(dtype=float)
X[:, 0] = X[:, 0] / X[:, 0].max()

# Response
y = data["quantity"].to_numpy(dtype=float).reshape(-1, 1)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
    shuffle=True
)

print("Training observations:", len(X_train))
print("Test observations:", len(X_test))


# ============================================================
# 3. BUILD NEURAL NETWORK
# ============================================================

# Input layer: 5 predictors
inputs = tf.keras.layers.Input(
    shape=(X_train.shape[1],),
    name="inputs"
)

# Hidden layer 1: 16 neurons
hidden1 = tf.keras.layers.Dense(
    units=16,
    activation="sigmoid",
    name="hidden1"
)(inputs)

# Hidden layer 2: 8 neurons
hidden2 = tf.keras.layers.Dense(
    units=8,
    activation="sigmoid",
    name="hidden2"
)(hidden1)

# Hidden layer 3: 4 neurons
hidden3 = tf.keras.layers.Dense(
    units=4,
    activation="sigmoid",
    name="hidden3"
)(hidden2)

# Output layer: 1 neuron
# Linear activation because quantity is continuous
output = tf.keras.layers.Dense(
    units=1,
    activation="linear",
    name="output"
)(hidden3)

# Create model
model = tf.keras.Model(
    inputs=inputs,
    outputs=output
)


# ============================================================
# 4. COMPILE MODEL
# ============================================================

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    metrics=["mae"]
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=100,              # DEVELOPMENT RUN
    validation_split=0.20,  # validation comes from training data
    shuffle=True,
    verbose=2
)


# ============================================================
# 6. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_train_pred = model.predict(
    X_train,
    verbose=0
)

y_test_pred = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

train_r2 = r2_score(
    y_train,
    y_train_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

train_mse = np.mean(
    (y_train - y_train_pred) ** 2
)

test_mse = np.mean(
    (y_test - y_test_pred) ** 2
)

train_mae = np.mean(
    np.abs(y_train - y_train_pred)
)

test_mae = np.mean(
    np.abs(y_test - y_test_pred)
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Training R2:", train_r2)
print("Test R2:", test_r2)

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)

print("Training MAE:", train_mae)
print("Test MAE:", test_mae)


# ============================================================
# 9. LEARNING CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


print("Actual quantity:")
print(pd.Series(y_test.flatten()).describe())

print("\nPredicted quantity:")
print(pd.Series(y_test_pred.flatten()).describe())

print("\nMean quantity:")
print(np.mean(y_train))