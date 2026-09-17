# ============================================================
# 4. BUILD THE NEURAL NETWORK
# ============================================================

# Input layer
# There are 5 predictor variables
inputs = tf.keras.layers.Input(
    shape=(X_train_scaled.shape[1],)
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

# Output layer
# One output because we are predicting quantity
output = tf.keras.layers.Dense(
    units=1,
    activation="linear",
    name="output"
)(hidden3)

# Create the model
model = tf.keras.Model(
    inputs=inputs,
    outputs=output
)


# ============================================================
# 5. COMPILE THE MODEL
# ============================================================

# Starting learning rate
learning_rate = 0.01

# Optimizer
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate
)

# (y - yhat)^2
# TensorFlow calls this Mean Squared Error (MSE)
model.compile(
    loss="mse",
    optimizer=optimizer
)

# View model architecture
model.summary()
