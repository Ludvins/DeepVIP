#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

from setup import experiment

import tensorflow as tf

X_train = np.loadtxt("data/SPGP_dist/train_inputs")
y_train = np.loadtxt("data/SPGP_dist/train_outputs")

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

experiment(X_train, y_train,
           n_layers = 1,
           structure=[10],
           epochs = 10000, 
           plotting = True, 
           fig_name="SPGP")
