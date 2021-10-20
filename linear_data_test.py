import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import Linear, BayesianLinearNN

import tensorflow as tf

tf.config.run_functions_eagerly(True)

######################################################
################### LINEAR DATASET ###################
######################################################


def f(x):
    return 3 * x + 5 + default_rng().standard_normal()


X_train = np.linspace(-5, 5, 200)
y_train = f(X_train)

y_mean = np.mean(y_train)
y_std = np.std(y_train)

X_train = X_train[..., np.newaxis]


######################################################
######################## LAYERS ######################
######################################################

# Regression coefficients
num_coeffs = 10

# Gaussian Likelihood
ll = Gaussian()

# Layers definition
rng = default_rng()
noise_sampler = lambda x: rng.standard_normal(size=x)

# Layers
layers = [
    VIPLayer(
        noise_sampler,
        BayesianLinearNN,
        num_regression_coeffs=num_coeffs,
        num_outputs=1,
        input_dim=1,
    ),
]

# DVIP
dvip = DVIP_Base(ll, layers, X_train.shape[0], y_mean=y_mean, y_std=y_std)

print(dvip.print_variables())

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

dvip.compile(opt)

dvip.fit(X_train, (y_train - y_mean) / y_std, epochs=50, batch_size=32)

print(dvip.print_variables())
mean, var = dvip(X_train)

plt.scatter(X_train, y_train, color="blue", label="Data")
plt.scatter(X_train, mean, color="red", label="Model fitting")
plt.legend()
plt.show()
