import tensorflow as tf
import numpy as np

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]  # (N, 28, 28, 1)
    x_test = x_test[..., None]
    return x_train, y_train, x_test, y_test
