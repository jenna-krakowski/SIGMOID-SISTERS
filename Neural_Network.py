# ============================================================
# GROUP ASSIGNMENT 1
# Incremental Learning on Large Data
# Feed-Forward Neural Network
# ============================================================


# ============================================================
# 1. IMPORT PACKAGES
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
import os
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. SETTINGS
# ============================================================

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Training settings
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# Moving-average window for learning curve
MOVING_WINDOW = 100


# ============================================================
# 3. LOAD DATA
# ============================================================

data = pd.read_csv("pricing.csv")

print("\n==============================")
print("DATA INFORMATION")
print("==============================")

print("Rows:", len(data))
print("Columns:", data.columns.tolist())

print("\nMissing values:")
print(data.isna().sum())

print("\nQuantity information:")
print("Mean:", data["quantity"].mean())
print("Median:", data["quantity"].median())
print("Maximum:", data["quantity"].max())


# ============================================================
# 4. DEFINE INPUTS AND RESPONSE
# ============================================================

# Input variables:
# sku
# price
# order
# duration
# category
#
# Response:
# quantity

feature_names = [
    "sku",
    "price",
    "order",
    "duration",
    "category"
]

X = data[feature_names].copy()

y = data["quantity"].to_numpy(
    dtype=np.float32
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

# 80% training
# 20% testing
#
# The test set is NOT used during training.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)


print("\n==============================")
print("TRAIN / TEST SPLIT")
print("==============================")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 6. SCALE INPUT VARIABLES
# ============================================================

# StandardScaler converts variables to approximately:
#
# 0  = training-set mean
# +1 = one standard deviation above mean
# -1 = one standard deviation below mean
#
# Scaling is useful for sigmoid neural networks because
# extremely large inputs can cause sigmoid saturation.
#
# IMPORTANT:
# Fit scaler ONLY using training data.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)


print("\nScaled training shape:")
print(X_train_scaled.shape)


# ============================================================
# 7. BUILD THE NEURAL NETWORK
# ============================================================

# Three hidden layers
# Sigmoid activation

inputs = tf.keras.layers.Input(
    shape=(X_train_scaled.shape[1],),
    name="input"
)


# Hidden Layer 1
hidden1 = tf.keras.layers.Dense(
    units=16,
    activation="sigmoid",
    name="hidden1"
)(inputs)


# Hidden Layer 2
hidden2 = tf.keras.layers.Dense(
    units=8,
    activation="sigmoid",
    name="hidden2"
)(hidden1)


# Hidden Layer 3
hidden3 = tf.keras.layers.Dense(
    units=4,
    activation="sigmoid",
    name="hidden3"
)(hidden2)


# Regression output
output = tf.keras.layers.Dense(
    units=1,
    activation="linear",
    name="output"
)(hidden3)


model = tf.keras.Model(
    inputs=inputs,
    outputs=output
)


# ============================================================
# 8. COMPILE MODEL
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="mse"
)


print("\n==============================")
print("MODEL ARCHITECTURE")
print("==============================")

model.summary()


# ============================================================
# 9. SET UP RAM TRACKING
# ============================================================

# Get the current Python process
process = psutil.Process(
    os.getpid()
)

# RAM being used before training begins
ram_before = (
    process.memory_info().rss
    / 1024**2
)

# Start peak RAM at current RAM usage
peak_ram = ram_before


print("\n==============================")
print("RAM BEFORE TRAINING")
print("==============================")

print(
    "RAM before training:",
    round(ram_before, 2),
    "MB"
)


# ============================================================
# 10. CREATE INCREMENTAL MINI-BATCH DATASET
# ============================================================

# Instead of updating the neural network using the
# entire training set at once, TensorFlow feeds the
# model mini-batches of 64 observations.
#
# After each mini-batch, the model's weights are updated.

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_train_scaled,
        y_train
    )
)

train_dataset = (
    train_dataset
    .shuffle(
        buffer_size=10000,
        seed=42
    )
    .batch(BATCH_SIZE)
)


# ============================================================
# 11. TRAIN INCREMENTALLY
# ============================================================

# Save MSE after every mini-batch
batch_losses = []

# Save total number of observations learned
instances_seen = []

total_instances = 0


print("\n==============================")
print("TRAINING")
print("==============================")


# Start training timer
start_time = time.perf_counter()


for epoch in range(EPOCHS):

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    epoch_losses = []


    for X_batch, y_batch in train_dataset:

        # ----------------------------------------------------
        # ONE INCREMENTAL MODEL UPDATE
        # ----------------------------------------------------

        batch_loss = model.train_on_batch(
            X_batch,
            y_batch
        )

        batch_loss = float(
            batch_loss
        )


        # ----------------------------------------------------
        # TRACK RAM AFTER THIS MINI-BATCH
        # ----------------------------------------------------

        current_ram = (
            process.memory_info().rss
            / 1024**2
        )

        # If this is the highest RAM observed so far,
        # save it as peak RAM.
        if current_ram > peak_ram:

            peak_ram = current_ram


        # ----------------------------------------------------
        # SAVE MINI-BATCH MSE
        # ----------------------------------------------------

        batch_losses.append(
            batch_loss
        )

        epoch_losses.append(
            batch_loss
        )


        # ----------------------------------------------------
        # COUNT INSTANCES LEARNED
        # ----------------------------------------------------

        batch_n = int(
            X_batch.shape[0]
        )

        total_instances += batch_n

        instances_seen.append(
            total_instances
        )


    # --------------------------------------------------------
    # EPOCH AVERAGE MSE
    # --------------------------------------------------------

    epoch_mean_loss = np.mean(
        epoch_losses
    )


    print(
        "Mean MSE:",
        round(
            epoch_mean_loss,
            4
        )
    )


# Stop timer
end_time = time.perf_counter()

training_time = (
    end_time - start_time
)


# ============================================================
# 12. FINAL RAM RESULTS
# ============================================================

ram_after = (
    process.memory_info().rss
    / 1024**2
)

peak_ram_increase = (
    peak_ram - ram_before
)


print("\n==============================")
print("TRAINING INFORMATION")
print("==============================")

print(
    "Training time:",
    round(training_time, 2),
    "seconds"
)

print(
    "RAM before training:",
    round(ram_before, 2),
    "MB"
)

print(
    "Peak RAM during training:",
    round(peak_ram, 2),
    "MB"
)

print(
    "Additional RAM used during training:",
    round(peak_ram_increase, 2),
    "MB"
)

print(
    "RAM after training:",
    round(ram_after, 2),
    "MB"
)


# ============================================================
# 13. TRAINING PREDICTIONS
# ============================================================

train_predictions = model.predict(
    X_train_scaled,
    batch_size=256,
    verbose=0
).flatten()


# ============================================================
# 14. TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test_scaled,
    batch_size=256,
    verbose=0
).flatten()


# ============================================================
# 15. TRAINING R-SQUARED
# ============================================================

train_r2 = r2_score(
    y_train,
    train_predictions
)


# ============================================================
# 16. TEST R-SQUARED
# ============================================================

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ============================================================
# 17. RMSE
# ============================================================

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


# ============================================================
# 18. FINAL MODEL RESULTS
# ============================================================

print("\n==============================")
print("FINAL MODEL RESULTS")
print("==============================")

print(
    "Training R²:",
    round(train_r2, 4)
)

print(
    "Test R²:",
    round(test_r2, 4)
)

print(
    "Training RMSE:",
    round(train_rmse, 4)
)

print(
    "Test RMSE:",
    round(test_rmse, 4)
)


# ============================================================
# 19. MOVING AVERAGE OF MSE
# ============================================================

loss_series = pd.Series(
    batch_losses
)

moving_mse = (
    loss_series
    .rolling(
        window=MOVING_WINDOW,
        min_periods=1
    )
    .mean()
)


# ============================================================
# 20. LEARNING CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    instances_seen,
    moving_mse
)

plt.xlabel(
    "Number of Instances Learned"
)

plt.ylabel(
    "Moving Average MSE"
)

plt.title(
    "Neural Network Learning Curve"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. PERMUTATION VARIABLE IMPORTANCE
# ============================================================

# Permutation importance:
#
# 1. Take one predictor
# 2. Randomly shuffle it
# 3. Make new predictions
# 4. Calculate the new MSE
#
# If MSE increases a lot after the variable is shuffled,
# the variable was important to the model.

print("\n==============================")
print("VARIABLE IMPORTANCE")
print("==============================")


# Original test MSE
baseline_mse = mean_squared_error(
    y_test,
    test_predictions
)


importance_results = []


for i, feature in enumerate(feature_names):

    # Copy test data
    X_permuted = X_test_scaled.copy()


    # Randomly scramble one predictor
    rng = np.random.default_rng(42)

    X_permuted[:, i] = rng.permutation(
        X_permuted[:, i]
    )


    # Predictions after scrambling predictor
    permuted_predictions = model.predict(
        X_permuted,
        batch_size=256,
        verbose=0
    ).flatten()


    # MSE after scrambling predictor
    permuted_mse = mean_squared_error(
        y_test,
        permuted_predictions
    )


    # Importance =
    # increase in MSE caused by permutation
    importance = (
        permuted_mse
        - baseline_mse
    )


    importance_results.append(
        importance
    )


importance_df = pd.DataFrame(
    {
        "Variable": feature_names,
        "Importance": importance_results
    }
)


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print(importance_df)


# ============================================================
# 22. VARIABLE IMPORTANCE PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    importance_df["Variable"],
    importance_df["Importance"]
)

plt.xlabel(
    "Variable"
)

plt.ylabel(
    "Increase in Test MSE After Permutation"
)

plt.title(
    "Permutation Variable Importance"
)

plt.tight_layout()

plt.show()


# ============================================================
# 23. PARTIAL DEPENDENCE FUNCTION
# ============================================================

# A partial dependence plot shows how the model's
# average predicted quantity changes as ONE predictor
# changes while averaging over the other observations.

def partial_dependence(
    model,
    X_data,
    feature_index,
    feature_name,
    grid_points=25
):


    # --------------------------------------------------------
    # USE 5TH TO 95TH PERCENTILE
    # --------------------------------------------------------

    low = np.percentile(
        X_data[:, feature_index],
        5
    )

    high = np.percentile(
        X_data[:, feature_index],
        95
    )


    grid = np.linspace(
        low,
        high,
        grid_points
    )


    pd_values = []


    # --------------------------------------------------------
    # CHANGE ONE FEATURE AT A TIME
    # --------------------------------------------------------

    for value in grid:

        X_temp = X_data.copy()

        X_temp[
            :,
            feature_index
        ] = value


        predictions = model.predict(
            X_temp,
            batch_size=256,
            verbose=0
        ).flatten()


        # Average predicted quantity
        pd_values.append(
            np.mean(predictions)
        )


    # --------------------------------------------------------
    # CREATE PDP
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        grid,
        pd_values
    )

    plt.xlabel(
        feature_name
        + " (Standardized)"
    )

    plt.ylabel(
        "Average Predicted Quantity"
    )

    plt.title(
        "Partial Dependence: "
        + feature_name
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 24. PARTIAL DEPENDENCE PLOTS
# ============================================================

# ------------------------------------------------------------
# PRICE
# ------------------------------------------------------------

partial_dependence(
    model,
    X_test_scaled,
    feature_index=1,
    feature_name="Price"
)


# ------------------------------------------------------------
# ORDER
# ------------------------------------------------------------

partial_dependence(
    model,
    X_test_scaled,
    feature_index=2,
    feature_name="Order"
)


# ------------------------------------------------------------
# DURATION
# ------------------------------------------------------------

partial_dependence(
    model,
    X_test_scaled,
    feature_index=3,
    feature_name="Duration"
)


# ============================================================
# 25. FINAL PRESENTATION SUMMARY
# ============================================================

print("\n==============================")
print("PRESENTATION SUMMARY")
print("==============================")

print(
    "Architecture: 16 -> 8 -> 4"
)

print(
    "Activation: Sigmoid"
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Training R²:",
    round(train_r2, 4)
)

print(
    "Test R²:",
    round(test_r2, 4)
)

print(
    "Training RMSE:",
    round(train_rmse, 4)
)

print(
    "Test RMSE:",
    round(test_rmse, 4)
)

print(
    "Training time:",
    round(training_time, 2),
    "seconds"
)

print(
    "RAM before training:",
    round(ram_before, 2),
    "MB"
)

print(
    "Peak RAM during training:",
    round(peak_ram, 2),
    "MB"
)

print(
    "Additional RAM used during training:",
    round(peak_ram_increase, 2),
    "MB"
)

print("\nVariable Importance:")
print(importance_df)