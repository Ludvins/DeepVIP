import numpy as np
from numpy.random import default_rng


def SPGP():
    X_train = np.loadtxt("data/SPGP_dist/train_inputs")
    y_train = np.loadtxt("data/SPGP_dist/train_outputs")
    X_test = np.loadtxt("data/SPGP_dist/test_inputs")

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = y_train[..., np.newaxis]

    return X_train, y_train, X_test, None


def synthetic():
    rng = default_rng(seed=0)

    def f(x):
        return (
            np.cos(5 * x) / (np.abs(x) + 1)
            + rng.standard_normal(x.shape) * 0.1
        )

    X_train = rng.standard_normal(300)
    y_train = f(X_train)
    X_test = rng.standard_normal(400) * 2
    y_test = f(X_test)

    X_train = X_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    return X_train, y_train, X_test, y_test


def boston():
    (X_train, y_train), (
        X_test,
        y_test,
    ) = tf.keras.datasets.boston_housing.load_data()

    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test
