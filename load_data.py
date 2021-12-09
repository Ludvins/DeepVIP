import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.model_selection import train_test_split

uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def test():

    X_test = np.linspace(-2, 2, 200)
    X_train = np.linspace(-1, 1, 50)
    y_train = np.sign(X_train)
    y_test = np.sign(X_test)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = y_train[..., np.newaxis]

    return X_train, y_train, X_test, y_test


def SPGP():
    X_train = np.loadtxt("data/SPGP_dist/train_inputs")
    y_train = np.loadtxt("data/SPGP_dist/train_outputs")
    X_test = np.loadtxt("data/SPGP_dist/test_inputs")

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = y_train[..., np.newaxis]

    X_test = np.linspace(-1, 8, 1000)
    X_test = X_test[..., np.newaxis]

    # X_train = (X_train - np.mean(X_train)) / np.std(X_train)

    return X_train, y_train, X_test, None


def synthetic():
    rng = default_rng(seed=0)

    def f(x):
        return np.cos(5 * x) / (np.abs(x) + 1)

    X_train = rng.standard_normal(300)
    y_train = f(X_train) + rng.standard_normal(X_train.shape) * 0.1
    X_test = rng.standard_normal(1000) * 3
    y_test = f(X_test)

    X_train = X_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    return X_train, y_train, X_test, y_test


def boston():
    print("Loading boston....")
    data_url = '{}{}'.format(uci_base, 'housing/housing.data')
    raw_df = pd.read_fwf(data_url, header=None).to_numpy()
    data = raw_df[:, :-1]
    target = raw_df[:, -1][..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.10,
                                                        random_state=42)

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    return ((X_train - X_mean) / X_std, y_train, (X_test - X_mean) / X_std,
            y_test)
