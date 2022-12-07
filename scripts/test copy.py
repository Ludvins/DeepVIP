from datetime import datetime
import itertools
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2
from functorch import jacrev

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

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)
fig = plt.gcf()

samples = f1(torch.tensor(X, dtype=args.dtype), 10000)

for f in samples[:100]:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")

print("Computing Moments...", end=" ")
mean = torch.mean(samples, dim=0).squeeze().detach().numpy()
cov_samples = torch.cov(samples.squeeze().T)
print("done")



L = np.linalg.cholesky(cov_samples.detach().numpy() + np.eye(mean.shape[0]) * 1e-6)
z = np.random.randn(30, mean.shape[0])


samples = mean + (L @ z.T).T
ax3.plot(X.flatten(), mean)
ax3.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov_samples.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov_samples.detach().numpy())),
    alpha=0.2,
)
for f in samples:
    ax3.plot(X.flatten(), f, alpha=0.3)

ax3.set_title("Samples from Gaussianized IP using Moments")
print("Computing Taylor...", end=" ")

w = f1.get_weights()

J = jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X), w)

J = torch.cat([torch.flatten(a, -2, -1) for a in J], dim=-1).transpose(-1, -2)
S = torch.exp(f1.get_std_params()) ** 2
cov_taylor = torch.einsum("nsd, s, msd -> nmd", J, S, J).squeeze(-1)


mean = f1.forward_mean(torch.tensor(X, dtype=args.dtype)).squeeze(-1).detach().numpy()

print("done")

L = np.linalg.cholesky(cov_taylor.detach().numpy() + np.eye(mean.shape[0]) * 1e-6)
z = np.random.randn(20, mean.shape[0])


samples = mean + (L @ z.T).T

ax5.plot(X.flatten(), mean)
ax5.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov_taylor.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov_taylor.detach().numpy())),
    alpha=0.2,
)
for f in samples:
    ax5.plot(X.flatten(), f, alpha=0.3)
ax5.set_title("Samples from Gaussianized IP using Taylor")

print("Computing GP...", end=" ")


GP_mean = f1.GP_mean(torch.tensor(X)).detach().numpy()
GP_cov = f1.GP_cov(torch.tensor(X))
GP_diag = np.sqrt(torch.diag(GP_cov).detach().numpy())
print("done")
    

vmin = np.min([np.min(cov_samples.detach().numpy()), np.min(cov_taylor.detach().numpy()), np.min(GP_cov.detach().numpy())])
vmax = np.max([np.max(cov_samples.detach().numpy()), np.max(cov_taylor.detach().numpy()), np.max(GP_cov.detach().numpy())])



pos = ax2.imshow(cov_samples.detach().numpy(), vmin=vmin, vmax=vmax)

fig.colorbar(pos, ax=ax2)
ax2.set_title("Covariance matrix Moments")

pos = ax6.imshow(cov_taylor.detach().numpy(), vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=ax6)
ax6.set_title("Covariance matrix Taylor")

pos = ax4.imshow(GP_cov.detach().numpy(), vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=ax4)
ax4.set_title("Covariance matrix GP")

plt.tight_layout()

plt.savefig("GP_taylor.pdf", dpi=1200, format="pdf")


plt.show()
