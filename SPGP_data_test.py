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

num_coeffs = 20

# Gaussian Likelihood
ll = Gaussian()

# Layers definition
rng = default_rng()
noise_sampler = lambda x: rng.standard_normal(size=x)

layers = [
    VIPLayer(
        noise_sampler,
        BayesianLinearNN,
        num_regression_coeffs=num_coeffs,
        num_outputs=1,
        input_dim=1,
    ),
]

dvip = DVIP_Base(ll, layers, X_train.shape[0], y_mean=y_mean, y_std=y_std)

print(dvip.print_variables())

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

dvip.compile(optimizer=opt)

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

dvip.fit(X_train, (y_train - y_mean) / y_std, epochs=500, batch_size=32)

print(dvip.print_variables())
mean, var = dvip.predict_y(X_train)

sort = np.argsort(X_train[:, 0])
fig, ax = plt.subplots()

ax.scatter(X_train, y_train, color="blue", label="Data")
ax.scatter(X_train, mean, color="red", label="Model fitting")

mean = mean.numpy()[sort, 0]
var = var.numpy()[sort, 0]

ax.fill_between(X_train[sort, 0], mean - 3 * var, mean + 3 * var, color="b", alpha=0.1)

plt.legend()
plt.show()
