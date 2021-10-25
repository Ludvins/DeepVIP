#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
import seaborn as sns

import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import BayesianNN, get_bn


import tensorflow as tf

def experiment(
    X,
    y,
    regression_coeffs=20,
    lr=0.001,
    epochs=1000,
    batch_size=None,
    structure=[10, 10],
    activation=tf.keras.activations.tanh,
    layers_shape = [],
    mean_function = None,
    eager = False,
    plotting=False,
    fig_name=None,
    show = False,
    verbose=1,
):

    # Set eager execution
    tf.config.run_functions_eagerly(eager)
    
    assert X.shape[0] == y.shape[0]

    y_mean = np.mean(y)
    y_std = np.std(y)

    if verbose > 0:
        print("Data information")
        print("Samples: ", X.shape[0])
        print("Features dimension: ", X.shape[1])
        print("Label dimension: ", y.shape[1])
        print("Labels mean value: ", y_mean)
        print("Labels deviation: ", y_std)
        print("press any key to continue...")
        input()

    if batch_size is None:
        batch_size = X.shape[0]

    # Gaussian Likelihood
    ll = Gaussian()

    # Layers definition
    rng = default_rng()
    noise_sampler = lambda x: rng.standard_normal(size=x)

    dims = [X.shape[1]] + layers_shape + [y.shape[1]]
    
    layers = [
        VIPLayer(
            noise_sampler,
            get_bn(structure, activation),
            num_regression_coeffs=regression_coeffs,
            num_outputs=_out,
            input_dim=_in,
            mean_function=mean_function
        )
        for _in, _out in zip(dims, dims[1:])
    ]
    
    dvip = DVIP_Base(ll, layers, X.shape[0], y_mean=y_mean, y_std=y_std)

    if verbose > 1:
        print("Initial variable values:")
        dvip.print_variables()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    dvip.compile(optimizer=opt)

    dvip.fit(X, (y - y_mean) / y_std, epochs=epochs, 
             batch_size=batch_size, 
             )

    if verbose > 1:
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
            + "-".join(str(a) for a in dims)
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
