#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

from setup import bnn_experiment

import tensorflow as tf

X_train = np.loadtxt("data/SPGP_dist/train_inputs")
y_train = np.loadtxt("data/SPGP_dist/train_outputs")

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

bnn_experiment(
    X_train,
    y_train,
    vip_layers=4,
    structure=[10],
    epochs=20000,
    plotting=True,
    fig_name="SPGP",
    show=True,
    verbose=0,
)
