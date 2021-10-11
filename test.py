import numpy as np
from numpy.random import default_rng


import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import BayesianLinearNN

import tensorflow as tf

tf.config.run_functions_eagerly(True)

X_train = np.loadtxt("data/SPGP_dist/train_inputs")
y_train = np.loadtxt("data/SPGP_dist/train_outputs")

y_mean = np.mean(y_train)
y_std = np.std(y_train)

X_train = X_train[..., np.newaxis]

print(X_train.shape)
print(y_train.shape)

num_coeffs = 10

# Gaussian Likelihood
ll = Gaussian()

# Layers definition
rng = default_rng()
noise_sampler = rng.standard_normal

layers = [
    VIPLayer(noise_sampler, BayesianLinearNN, num_regression_coeffs=num_coeffs,
             num_outputs=1, input_dim=1),
]

dvip = DVIP_Base(ll, layers, X_train.shape[0], num_samples=10, y_mean=y_mean, y_std = y_std)

print(dvip.print_variables())

dvip.compile(optimizer="adam")

dvip.fit(X_train, (y_train - y_mean)/y_std, epochs=200, batch_size=50)

print(dvip.print_variables())
mean, var = dvip.predict_f(X_train)

plt.scatter(X_train, y_train, color = "blue", label = "Data")
plt.scatter(X_train, mean, color = "red", label = "Model fitting")
plt.legend()
plt.show()
