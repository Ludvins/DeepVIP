from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2
import gpytorch

sys.path.append(".")

from src.fvi import FVI, FVI2, FVI3, FVI4, FVI5
from src.layers_init import init_layers, init_layers_tvip
from src.layers import TVIPLayer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

#args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = torch.device("cpu")
torch.manual_seed(args.seed)
args.dtype = torch.float32

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit='points', seed = args.seed)[0]

f = GP(
    input_dim=1,
    output_dim=1,
    inner_layer_dim=200,
    kernel_amp=np.exp(4.83368164),
    kernel_length=np.exp(-0.42975142),
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

X = torch.tensor(test_dataset.inputs, dtype =args.dtype)

class Kernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **args):
        return f.GP_cov(x1, x2)

class Mean(gpytorch.means.Mean):                                                                                                                                                                        

    def forward(self, x):
        return f.GP_mean(x).T

X = torch.tensor(train_dataset.inputs).to(args.dtype)
X_test = torch.tensor(test_dataset.inputs).to(args.dtype)
K_t = f.GP_cov(X_test, X_test)
K_x = f.GP_cov(X, X)
K_xt = f.GP_cov(X, X_test)

print(K_t.shape)
print(K_x.shape)
print(K_xt.shape)

jitter = np.exp(-1.74978542)

mean = f.GP_mean(X_test) + K_xt.T @ torch.inverse(K_x + jitter * torch.eye(K_x.shape[0]))\
    @ (torch.tensor(train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean).to(args.dtype) -  f.GP_mean(X ))
pred = K_t + jitter * torch.eye(K_t.shape[0]) - K_xt.T @ torch.inverse(K_x + jitter * torch.eye(K_x.shape[0])) @ K_xt
print(mean.shape)
print(pred.shape)
var = torch.diagonal(pred)


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))



x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
plt.scatter(train_dataset.inputs, y, label="Training dataset", s=5)


plt.plot(test_dataset.inputs.flatten(), mean.detach().numpy().flatten())
plt.fill_between(test_dataset.inputs.flatten(),
                 mean.detach().numpy().flatten() - 3*var.detach().numpy().flatten(),
                 mean.detach().numpy().flatten() + 3*var.detach().numpy().flatten(),
                 alpha = 0.2
                 )
plt.show()
