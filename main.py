# The aim is to mimic keras' dense layer

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_layers
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed for tf.random", type=int, default=0)

args = parser.parse_args()

SEED: int = args.seed

tf.random.set_seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

# Define the model architecture.
def get_model(model_type: str):
    """Options: "dense', or 'custom_dense'."""
    if "custom" not in model_type:
        return keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28)),
                keras.layers.Flatten(),
                keras.layers.Dense(10),
                keras.layers.Dense(10),
                keras.layers.Dense(10),
            ]
        )
    else:
        return keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28)),
                keras.layers.Flatten(),
                custom_layers.Dense(10),
                custom_layers.Dense(10),
                custom_layers.Dense(10),
            ]
        )


# Train the base model
tf.random.set_seed(SEED)
base_model = get_model("dense")
base_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
base_model.fit(train_images, train_labels, epochs=1, validation_split=0.1, verbose=1)

base_model.summary()

# Train the custom dense layer model
tf.random.set_seed(SEED)
custom_model = get_model("custom_dense")
custom_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
custom_model.fit(train_images, train_labels, epochs=1, validation_split=0.1, verbose=1)

_, base_model_accuracy = base_model.evaluate(test_images, test_labels, verbose=0)
_, custom_model_accuracy = custom_model.evaluate(test_images, test_labels, verbose=0)

print("Base test accuracy:", base_model_accuracy)
print("Custom test accuracy:", custom_model_accuracy)

# Run test dataset on custom, and TFLite models
base_output: np.ndarray = base_model.predict(test_images)
custom_output: np.ndarray = custom_model.predict(test_images)
base_output = base_output.flatten()
custom_output = custom_output.flatten()
utils.output_stats(base_output, custom_output, "Base vs Custom", 1e-6, SEED)
