
# ==========================================
# GROUP ASSIGNMENT 2: MODEL EXPERIMENT TEMPLATE
# ==========================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================
# 1. EXPERIMENT SETTINGS
# Change these for each trial
# ==========================================

experiment_name = "Baseline"

# Predictors
use_sku = True

# Neural network architecture
hidden_layers = [16, 8, 4]
activation = "sigmoid"

# Training settings
optimizer_name = "adam"
learning_rate = 0.001
batch_size = 64
epochs = 100


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

pricing = pd.read_csv("pricing.csv")

predictors = ["price", "order", "duration", "category"]

if use_sku:
    predictors = ["sku"] + predictors

X = pricing[predictors].to_numpy()
y = pricing["quantity"].to_numpy()

# Split into 70% training, 15% validation, 15% testing

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=15/85, random_state=42
)


# ==========================================
# 3. STANDARDIZE INPUTS
# Fit scaler ONLY on training data
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ==========================================
# 4. BUILD NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential()

# Input layer
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

# Hidden layers
for neurons in hidden_layers:
    model.add(tf.keras.layers.Dense(
        neurons,
        activation=activation
    ))

# Output layer
model.add(tf.keras.layers.Dense(1, activation="linear"))


# ==========================================
# 5. SELECT OPTIMIZER
# ==========================================

if optimizer_name == "adam":
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

elif optimizer_name == "sgd":
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )

elif optimizer_name == "momentum":
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9
    )

elif optimizer_name == "nesterov":
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True
    )

elif optimizer_name == "adagrad":
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate
    )

elif optimizer_name == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate
    )

else:
    raise ValueError("Unknown optimizer")


# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

training_time = time.time() - start_time


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

train_pred = model.predict(X_train, verbose=0).flatten()
val_pred = model.predict(X_val, verbose=0).flatten()
test_pred = model.predict(X_test, verbose=0).flatten()


def calculate_metrics(actual, predicted):
    return {
        "R2": r2_score(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted)
    }


train_results = calculate_metrics(y_train, train_pred)
val_results = calculate_metrics(y_val, val_pred)
test_results = calculate_metrics(y_test, test_pred)


# ==========================================
# 9. PRINT RESULTS
# ==========================================

print("\n========== EXPERIMENT RESULTS ==========")
print("Experiment:", experiment_name)
print("SKU included:", use_sku)
print("Hidden layers:", hidden_layers)
print("Activation:", activation)
print("Optimizer:", optimizer_name)
print("Learning rate:", learning_rate)
print("Batch size:", batch_size)
print("Epochs:", epochs)

print("\nTraining Results:", train_results)
print("Validation Results:", val_results)
print("Testing Results:", test_results)

print("\nTraining time:", round(training_time, 2), "seconds")


# ==========================================
# 10. PLOT LEARNING CURVE
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(experiment_name + " - Learning Curve")
plt.legend()
plt.show()


# ==========================================
# 11. SAVE EXPERIMENT RESULTS
# ==========================================

results_row = {
    "Experiment": experiment_name,
    "SKU": use_sku,
    "Hidden Layers": str(hidden_layers),
    "Activation": activation,
    "Optimizer": optimizer_name,
    "Learning Rate": learning_rate,
    "Batch Size": batch_size,
    "Epochs": epochs,
    "Train RMSE": train_results["RMSE"],
    "Validation RMSE": val_results["RMSE"],
    "Test RMSE": test_results["RMSE"],
    "Train R2": train_results["R2"],
    "Validation R2": val_results["R2"],
    "Test R2": test_results["R2"],
    "Training Time": training_time
}

results_df = pd.DataFrame([results_row])

results_df.to_csv(
    "experiment_results.csv",
    mode="a",
    header=not pd.io.common.file_exists("experiment_results.csv"),
    index=False
)
