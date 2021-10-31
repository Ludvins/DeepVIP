#!/usr/bin/env python3

from src.layers import VIPLayer
import numpy as np
import tensorflow as tf
from src.generative_models import get_bn
from numpy.random import default_rng


def init_layers(
    X,
    Y,
    inner_dims,
    regression_coeffs=20,
    structure=[10, 10],
    activation=tf.keras.activations.tanh,
    noise_sampler=None,
):

    if noise_sampler is None:

        def noise_sampler(x):
            return default_rng().standard_normal(size=x)

    if isinstance(inner_dims, int):
        dims = np.concatenate(
            ([X.shape[1]], np.ones(inner_dims, dtype=int) * Y.shape[1])
        )
    else:
        dims = [X.shape[1]] + inner_dims + [Y.shape[1]]

    layers = []

    X_running = np.copy(X)
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):

        if dim_in == dim_out:

            def mf(x):
                return x

        else:
            if dim_in > dim_out:
                _, _, V = np.linalg.svd(X_running, full_matrices=False)

                def mf(x):
                    return x @ V[:dim_out, :].T

            else:
                raise NotImplementedError

        X_running = mf(X_running)

        layers.append(
            VIPLayer(
                noise_sampler,
                get_bn(structure, activation),
                num_regression_coeffs=regression_coeffs,
                num_outputs=dim_out,
                input_dim=dim_in,
                mean_function=mf,
            )
        )

    layers.append(
        VIPLayer(
            noise_sampler,
            get_bn(structure, activation),
            num_regression_coeffs=regression_coeffs,
            num_outputs=dims[-1],
            input_dim=dims[-2],
            mean_function=None,
        )
    )

    return layers
