from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2
from functorch import jacrev, hessian, jacfwd, vmap

sys.path.append(".")

from src.fvi import FVI
from src.layers_init import init_layers, init_layers_tvip
from src.layers import TVIPLayer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

# args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

Z = kmeans2(train_dataset.inputs, 10, minit="points", seed=args.seed)[0]

f1 = BayesianNN(
    input_dim=1,
    structure=args.bnn_structure,
    activation=args.activation,
    output_dim=1,
    layer_model=BayesLinear,
    dropout=args.dropout,
    fix_mean = args.fix_mean,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)



f1 = GP(
    input_dim=1,
    output_dim=1,
    inner_layer_dim=20,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)


X = test_dataset.inputs
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
fig = plt.gcf()

samples = f1(torch.tensor(X, dtype=args.dtype), 200)
for f in samples:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")

mean = torch.mean(samples, dim=0).squeeze().detach().numpy()
cov = torch.cov(samples.squeeze().T)

ax2.plot(X.flatten(), mean, alpha = 0.5)
ax2.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov.detach().numpy())),
    alpha=0.1,
)
ax4.plot(X.flatten(), mean, alpha = 0.9, label = "GP mean")
ax4.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov.detach().numpy())),
    alpha=0.2,
    label = "GP std"
)


ax2.set_title("GP using Moments")





n = 20
w = f1.sample_weights(n)
w_mean = f1.get_weights()
diff = w - w_mean


J = torch.stack(
    [jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), a) for a in w]
    )

S = torch.exp(f1.get_std_params()) ** 2
cov = torch.einsum("ands, s, bnds -> abnd", J, S, J).squeeze(-1)
mean = f1.forward_weights(torch.tensor(X, dtype=args.dtype), w)\
    + torch.einsum("ands, as -> and", J, diff)

cov_diag = torch.diagonal(cov, dim1 = 0, dim2 = 1).T

mean = mean.squeeze(-1).detach().cpu().numpy()

for i in range(mean.shape[0]):

    ax3.plot(X.flatten(), mean[i])
    ax3.fill_between(
        X.flatten(),
        mean[i] - 2 * np.sqrt(cov_diag[i].detach().numpy()),
        mean[i] + 2 * np.sqrt(cov_diag[i].detach().numpy()),
        alpha=0.2,
    )
ax3.set_title("Linearized GP at random parameter values")



mean = np.mean(mean, axis = 0)
cov = np.sum(cov.detach().numpy(), axis = (0, 1))/(cov.shape[0]**2)
ax4.plot(X.flatten(), mean, alpha = 0.9, label = "Ensemble Mean")
ax4.fill_between(
        X.flatten(),
        mean - 2 * np.sqrt(cov),
        mean + 2 * np.sqrt(cov),
        alpha=0.2,
        label = "Ensemble std"
   )

ax4.set_title("Distributions")

J = jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), w_mean)
cov = torch.einsum("nds, s, nds -> nd", J, S, J).squeeze(-1).detach().numpy()
mean = f1.forward_weights(torch.tensor(X, dtype=args.dtype), w_mean).squeeze(-1).detach().cpu().numpy()
ax4.plot(X.flatten(), mean, alpha = 0.9, label = "Taylor Mean")
ax4.fill_between(
        X.flatten(),
        mean - 2 * np.sqrt(cov),
        mean + 2 * np.sqrt(cov),
        alpha=0.2,
        label = "Taylor std"
   )
ax4.legend()

ylims = ax1.get_ylim()

ax1.set_ylim(ylims)
ax2.set_ylim(ylims)
ax3.set_ylim(ylims)
ax4.set_ylim(ylims)

plt.tight_layout()
plt.savefig("Prior distribution.pdf", format="pdf")
plt.show()
