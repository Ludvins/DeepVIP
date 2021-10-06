import numpy as np
from numpy.random import default_rng


import matplotlib.pyplot as plt

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers import VIPLayer
from src.generative_models import Linear

X_train = np.loadtxt("data/SPGP_dist/train_inputs")
y_train = np.loadtxt("data/SPGP_dist/train_outputs")

X_train = X_train[..., np.newaxis]

print(X_train.shape)
print(y_train.shape)

# Gaussian Likelihood
ll = Gaussian()

# Layers definition
rng = default_rng()
noise_sampler = rng.standard_normal
net = Linear(noise_sampler, 10, input_dim=1)

layers = [
    VIPLayer(net, layer_noise=0.1, num_regression_coeffs=10,
             num_outputs=1, input_dim=1)
]

dvip = DVIP_Base(ll, layers, X_train.shape[0])

print(dvip.trainable_variables)

dvip.compile(optimizer="adam")

dvip.fit(X_train, y_train, epochs=10, batch_size=50)

mean, var = dvip.predict_f(X_train)

plt.scatter(X_train, y_train, color = "blue", label = "Data")
plt.scatter(X_train, mean, color = "red", label = "Model fitting")
plt.legend()
plt.show()
