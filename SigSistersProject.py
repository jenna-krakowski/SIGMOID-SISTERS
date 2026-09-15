# ============================================================
# IMPORT PACKAGES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import tracemalloc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Makes results more reproducible
np.random.seed(1)
tf.random.set_seed(1)


# ============================================================
# 1. SETTING UP THE DATA
# ============================================================

# Read in the dataset
data = pd.read_csv("pricing.csv")


# Keep only the variables needed for the assignment
data = data[
    ["sku", "price", "order", "duration", "category", "quantity"]
].copy()


# ============================================================
# CONVERT CATEGORICAL VARIABLES
# ============================================================

# Convert sku and category to numeric codes
data["sku"] = pd.Categorical(data["sku"]).codes
data["category"] = pd.Categorical(data["category"]).codes


# Remove rows with missing values
data = data.dropna()


# ============================================================
# CREATE X AND y
# ============================================================

# X contains the 5 predictor variables
X = data[
    ["sku", "price", "order", "duration", "category"]
].to_numpy(dtype=float)


# y contains the response variable
y = data["quantity"].to_numpy(dtype=float).reshape(-1, 1)


# ============================================================
# 2. CREATE TRAINING AND TEST DATA
# ============================================================

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
    shuffle=True
)


# ============================================================
# 3. STANDARDIZE THE PREDICTORS
# ============================================================

# Standardize X variables so they are on similar scales
scaler = StandardScaler()

# Learn scaling from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same scaling to test data
X_test_scaled = scaler.transform(X_test)


