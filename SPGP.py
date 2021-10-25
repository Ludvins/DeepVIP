#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

from setup import experiment

import tensorflow as tf

X_train = np.loadtxt("data/SPGP_dist/train_inputs")
y_train = np.loadtxt("data/SPGP_dist/train_outputs")

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

mean_f = lambda x: x

layers = [[], [1], [1,1], [1,1,1]]
bns = [[10], [10, 10]]

for l in layers:
    for b in bns:
        
        experiment(X_train, y_train,
                layers_shape=l,
                structure=b,
                epochs = 20000, 
                plotting = True, 
                fig_name="SPGP",
                show = False,
                verbose = 0)
         
        experiment(X_train, y_train,
                layers_shape=l,
                structure=b,
                epochs = 20000, 
                mean_function=mean_f,
                plotting = True, 
                fig_name="SPGP_mean",
                show = False,
                verbose = 0)

