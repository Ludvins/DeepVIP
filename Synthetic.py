import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import BayesianNN, get_bn

import tensorflow as tf

from setup import experiment


######################################################
################### LINEAR DATASET ###################
######################################################

rng = default_rng(seed = 2021)

def f(x):
    return np.cos(5*x)/(np.abs(x) + 1) + rng.standard_normal()*0.1


X_train = default_rng().standard_normal(300)
y_train = f(X_train)

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

experiment(X_train, y_train,
           n_layers=2,
           structure = [10],
           verbose = 0,
           epochs = 10000, 
           plotting = True,
           fig_name="Synthetic")
