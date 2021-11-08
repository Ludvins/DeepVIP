#!/usr/bin/env python3

from src.layers import VIPLayer
import numpy as np
import tensorflow as tf
from numpy.random import default_rng
from src.generative_models import GaussianSampler, BayesianNN


class LinearProjection:
    def __init__(self, matrix):
        self.P = matrix

    def __call__(self, inputs):
        return inputs @ self.P.T


def init_layers(X,
                Y,
                inner_dims,
                regression_coeffs=20,
                structure=[10, 10],
                activation=tf.keras.activations.tanh,
                noise_sampler=None,
                seed=0):

    if noise_sampler is None:
        noise_sampler = GaussianSampler(0)

    if isinstance(inner_dims, (int, np.integer)):
        dims = np.concatenate(
            ([X.shape[1]], np.ones(inner_dims, dtype=int) * Y.shape[1]))
    else:
        dims = [X.shape[1]] + inner_dims + [Y.shape[1]]

    layers = []
    X_running = np.copy(X)
    for (i, (dim_in, dim_out)) in enumerate(zip(dims[:-1], dims[1:])):
        if i == len(dims) - 2:
            mf = None

        elif dim_in == dim_out:

            mf = LinearProjection(np.identity(n=dim_in))

        elif dim_in > dim_out:
            _, _, V = np.linalg.svd(X_running, full_matrices=False)

            mf = LinearProjection(V[:dim_out, :])
            X_running = mf(X_running)

        else:
            raise NotImplementedError("Dimensionality augmentation is not"
                                      " handled currently.")

        bayesian_network = BayesianNN(
            noise_sampler=noise_sampler,
            num_samples=regression_coeffs,
            input_dim=dim_in,
            structure=structure,
            activation=activation,
            num_outputs=dim_out,
            seed=seed,
        )
        layers.append(
            VIPLayer(
                bayesian_network,
                num_regression_coeffs=regression_coeffs,
                num_outputs=dim_out,
                input_dim=dim_in,
                mean_function=mf,
            ))

    return layers
