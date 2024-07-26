import keras.utils
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

# Set seed for reproducability
keras.utils.set_random_seed(42)

# Use the Boston housing data set
boston_data = pd.read_csv("https://evermann.ca/busi4720/boston.csv")

# Separate features and targets
boston_features = boston_data[['rm', 'tax', 'age']]
boston_labels = boston_data['medv']

# Linear regression model
norm_boston_model=keras.models.Sequential([
    keras.layers.Input(shape=(3,), dtype=tf.float32),
    keras.layers.Dense(1, activation=None)
])

# Define early stopping
stop_callback = keras.callbacks.EarlyStopping()
# Define MSE loss
norm_boston_model.compile(
    loss = tf.keras.losses.MeanSquaredError())
# Fit model to data
norm_boston_model.fit(
    boston_features, boston_labels,
    epochs=100, validation_split=0.33,
    callbacks=[stop_callback])

# Save model for use in Keras
norm_boston_model.save('norm.boston.model.trained.save')
# Export model for use in TF Serving
norm_boston_model.export('norm.boston.model.trained.export')
# Convert model for use in TFJS
tfjs.converters.save_keras_model(norm_boston_model, 'norm.boston.model.trained.tjfs')

