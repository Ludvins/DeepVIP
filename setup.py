#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
import seaborn as sns

import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import BayesianNN, get_bn
from src.layers_init import init_layers

import tensorflow as tf

seed = 123


def bnn_experiment(
    X,
    y,
    regression_coeffs=20,
    lr=0.001,
    epochs=1000,
    batch_size=None,
    structure=[10, 10],
    activation=tf.keras.activations.tanh,
    vip_layers=1,
    mean_function=None,
    eager=False,
    plotting=False,
    fig_name=None,
    show=False,
    verbose=1,
):

    # Set eager execution
    tf.config.run_functions_eagerly(eager)

    if X.shape[0] != y.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    # Compute data information
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    y_mean = np.mean(y)
    y_std = np.std(y)

    if verbose > 0:
        print("Data information")
        print("Samples: ", n_samples)
        print("Features dimension: ", input_dim)
        print("Label dimension: ", output_dim)
        print("Labels mean value: ", y_mean)
        print("Labels deviation: ", y_std)
        print("press any key to continue...")
        input()

    # If no batch size is set, use the full dataset
    if batch_size is None:
        batch_size = X.shape[0]

    # Gaussian Likelihood
    ll = Gaussian()

    # Layers definition
    rng = default_rng(seed)

    def noise_sampler(x):
        return rng.standard_normal(size=x)

    layers = init_layers(
        X, y, vip_layers, regression_coeffs, structure, activation, noise_sampler
    )

    dvip = DVIP_Base(ll, layers, input_dim, y_mean=y_mean, y_std=y_std)

    if verbose > 1:
        print("Initial variable values:")
        dvip.print_variables()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    dvip.compile(optimizer=opt)

    dvip.fit(
        X,
        (y - y_mean) / y_std,
        epochs=epochs,
        batch_size=batch_size,
    )

    if verbose > 1:
        print("Learned variable values.")
        dvip.print_variables()

    if plotting and X.shape[1] == 1:

        mean, var = dvip.predict_y(X)
        mean = mean * y_std + y_mean

        sort = np.argsort(X[:, 0])
        _, ax = plt.subplots()

        ax.scatter(X, y, color="blue", label="Data")
        ax.scatter(X, mean, color="red", label="Model fitting")

        mean = mean.numpy()[sort, 0]
        std = np.sqrt(var.numpy()[sort, 0])

        ax.fill_between(
            X[sort, 0], mean - 3 * std, mean + 3 * std, color="b", alpha=0.1
        )

        plt.legend()
        plt.savefig(
            "plots/"
            + fig_name
            + "_layers="
            + str(vip_layers)
            + "_bnn="
            + "-".join(str(a) for a in structure)
            + "_epochs="
            + str(epochs)
            + "_batchsize="
            + str(batch_size)
            + ".png"
        )
        if show:
            plt.show()
