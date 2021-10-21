#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng


import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import BayesianNN, get_bn

from datetime import datetime


import tensorflow as tf

# tf.config.run_functions_eagerly(True)


def experiment(
    X_train,
    y_train,
    regression_coeffs=20,
    lr=0.001,
    epochs=1000,
    batch_size=None,
    verbose=1,
    structure=[10, 10],
    activation=tf.keras.activations.tanh,
    n_layers=1,
    plotting=False,
    fig_name=None,
):

    assert X_train.shape[0] == y_train.shape[0]

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    if verbose > 0:
        print("Data information")
        print("Samples: ", X_train.shape[0])
        print("Features dimension: ", X_train.shape[1])
        print("Label dimension: ", y_train.shape[1])
        print("Labels mean value: ", y_mean)
        print("Labels deviation: ", y_std)
        print("press any key to continue...")
        input()

    if batch_size is None:
        batch_size = X_train.shape[0]

    # Gaussian Likelihood
    ll = Gaussian()

    # Layers definition
    rng = default_rng()
    noise_sampler = lambda x: rng.standard_normal(size=x)

    layers = [
        VIPLayer(
            noise_sampler,
            get_bn(structure, activation),
            num_regression_coeffs=regression_coeffs,
            num_outputs=y_train.shape[1],
            input_dim=X_train.shape[1],
        )
        for _ in range(n_layers)
    ]

    dvip = DVIP_Base(ll, layers, X_train.shape[0], y_mean=y_mean, y_std=y_std)

    if verbose > 1:
        print("Initial variable values:")
        dvip.print_variables()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    dvip.compile(optimizer=opt)

    dvip.fit(X_train, (y_train - y_mean) / y_std, epochs=epochs, 
             batch_size=batch_size, 
             )

    if verbose > 1:
        dvip.print_variables()

    if plotting and X_train.shape[1] == 1:

        mean, var = dvip.predict_y(X_train)
        mean = mean * y_std + y_mean
        
        sort = np.argsort(X_train[:, 0])
        _, ax = plt.subplots()
        
        

        ax.scatter(X_train, y_train, color="blue", label="Data")
        ax.scatter(X_train, mean, color="red", label="Model fitting")
        
        
        
        mean = mean.numpy()[sort, 0]
        std = np.sqrt(var.numpy()[sort, 0])

        ax.fill_between(
            X_train[sort, 0], mean - 3 * std, mean + 3 * std, color="b", alpha=0.1
        )

        plt.legend()
        plt.savefig(
            "plots/"
            + fig_name
            + "_nlayers="
            + str(n_layers)
            + "_bnn="
            + "-".join(str(a) for a in structure)
            + "_epochs="
            + str(epochs)
            + "_batchsize="
            + str(batch_size)
            + ".png"
        )
