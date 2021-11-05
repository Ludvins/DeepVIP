#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.generative_models import GaussianSampler

import tensorflow as tf


def check_data(X, y):
    if X.shape[0] != y.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    # Compute data information
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    y_mean = np.mean(y)
    y_std = np.std(y)

    return n_samples, input_dim, output_dim, y_mean, y_std


def show_unidimensional_results(X, y, mean, var, path=None, fig_title=None, show=False):

    sort = np.argsort(X[:, 0])
    _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    ax[0].scatter(
        X, y, color="blue", s=0.5, alpha=0.7, label="VIP - training noisy sample"
    )

    mean = mean.numpy()[sort, 0]
    std = np.sqrt(var.numpy()[sort, 0])

    ax[0].plot(X[sort], mean, color="teal", label="VIP - interpolation mean")

    ax[0].fill_between(
        X[sort, 0], mean - 2 * std, mean + 2 * std, color="teal", alpha=0.3
    )

    plt.legend()
    ax[0].set_title(fig_title)
    ax[1].fill_between(X[sort, 0], np.zeros(len(std)), std, color="teal", alpha=0.3)

    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()


def experiment(
    X_train,
    y_train,
    regression_coeffs=20,
    lr=0.001,
    epochs=1000,
    batch_size=None,
    structure=[10, 10],
    activation=tf.keras.activations.tanh,
    vip_layers=1,
    verbose=0,
    seed=0,
        eager_execution = False,
):

    # Set eager execution
    tf.config.run_functions_eagerly(eager_execution)

    n_samples, input_dim, output_dim, y_mean, y_std = check_data(X_train, y_train)

    # If no batch size is set, use the full dataset
    if batch_size is None:
        batch_size = n_samples

    # Gaussian Likelihood
    ll = Gaussian()

    # Define the noise sampler
    noise_sampler = GaussianSampler(seed)

    # Get VIP layers
    layers = init_layers(
        X_train,
        y_train,
        vip_layers,
        regression_coeffs,
        structure,
        activation,
        noise_sampler,
    )

    dvip = DVIP_Base(ll, layers, input_dim, y_mean=y_mean, y_std=y_std)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    dvip.compile(optimizer=opt)

    dvip.fit(
        X_train,
        (y_train - y_mean) / y_std,  # Provide normalized outputs
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return dvip
