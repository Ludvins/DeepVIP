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

from src.fvi import FVI, FVI2, FVI3, FVI4
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

Z = kmeans2(train_dataset.inputs, 5, minit='points', seed = args.seed)[0]

f = GP(
    input_dim=1,
    output_dim=1,
    inner_layer_dim=20,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

X = torch.tensor(test_dataset.inputs, dtype =args.dtype)


# def GP_cov( x1, x2, **args):
#     samples_1 = f(torch.cat([x1, x2]), 100000)
#     mean = torch.mean(samples_1, dim=0)
#     m = samples_1 - mean
#     cov = torch.einsum("snd, smd ->nmd", m, m) / (m.shape[0])
#     return cov[:x1.shape[0]:, x1.shape[0]:].squeeze(-1)


# def GP_mean(x):
    
#     samples = f(x, 50000)
#     return torch.mean(samples, dim = 0)


def GP_cov( x1, x2, **args):
    from functorch import jacrev, jacfwd

    w = f.get_weights()

    J = jacrev(f.forward_weights, argnums=1)(torch.concat([x1, x2], 0), w)

    J = (
        torch.cat(
            [
                torch.flatten(a, 2, -1) if len(a.shape) > 2 else a.unsqueeze(-1)
                for a in J
            ],
            dim=-1,
        )
        .transpose(-1, -2)
    )

    S = torch.exp(f.get_std_params()) ** 2

    cov = torch.einsum("nsd, s, msd -> nmd", J, S, J)
    return cov[:x1.shape[0]:, x1.shape[0]:].squeeze(-1)


def GP_mean(x):
    a = f.forward_mean(x)
    return a
    

# def GP_cov( x1, x2, **args):
#     return f.GP_cov(x1, x2)


# def GP_mean(x):
#     a = f.GP_mean(x)
#     return a
    


X = torch.tensor(train_dataset.inputs).to(args.dtype)
X_test = torch.tensor(test_dataset.inputs).to(args.dtype)

K_t = GP_cov(X_test, X_test)
K_x = GP_cov(X, X)
K_xt = GP_cov(X, X_test)

print(K_t)

jitter = np.exp(-1.74978542)

mean = GP_mean(X_test) + K_xt.T @ torch.inverse(K_x + jitter * torch.eye(K_x.shape[0]))\
    @ (torch.tensor(train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean).to(args.dtype) -  GP_mean(X ))
pred = K_t + jitter * torch.eye(K_t.shape[0]) - K_xt.T @ torch.inverse(K_x + jitter * torch.eye(K_x.shape[0])) @ K_xt
print(mean.shape)
print(pred.shape)
var = torch.diagonal(pred)


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))



x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
plt.scatter(train_dataset.inputs, y, label="Training dataset", s=5, color = "darkorange")


plt.plot(test_dataset.inputs.flatten(), mean.detach().numpy().flatten(), linewidth=3)
plt.fill_between(test_dataset.inputs.flatten(),
                 mean.detach().numpy().flatten() - 2*np.sqrt(var.detach().numpy().flatten()),
                 mean.detach().numpy().flatten() + 2*np.sqrt(var.detach().numpy().flatten()),
                 alpha = 0.2
                 )
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-3.2, 3.2)
plt.savefig("Exact_GP_taylor.pdf", format = "pdf")
plt.show()
