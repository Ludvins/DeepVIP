#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
import seaborn as sns
from setup import experiment
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

y_train = y_train[..., np.newaxis]
y_test = y_test[..., np.newaxis]

tf.config.run_functions_eagerly(True)

experiment(X_train, y_train,
           layers_shape = [10],
           structure=[10, 10],
           epochs = 10000, 
           plotting = True, 
           fig_name="boston")
