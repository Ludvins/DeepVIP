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
    inner_layer_dim=200,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)


X = test_dataset.inputs
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 3, 7)
ax6 = plt.subplot(3, 3, 8)
ax7 = plt.subplot(3, 3, 9)
fig = plt.gcf()

samples = f1(torch.tensor(X, dtype=args.dtype), 500)
for f in samples[:100]:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")

mean = torch.mean(samples, dim=0).squeeze().detach().numpy()
cov = torch.cov(samples.squeeze().T)

pos = ax5.imshow(cov.detach().numpy())

fig.colorbar(pos, ax=ax5)
ax5.set_title("Covariance matrix Moments")


L = np.linalg.cholesky(cov.detach().numpy() + np.eye(mean.shape[0]) * 1e-6)
z = np.random.randn(30, mean.shape[0])


samples = mean + (L @ z.T).T
ax2.plot(X.flatten(), mean)
ax2.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov.detach().numpy())),
    alpha=0.2,
)
for f in samples:
    ax2.plot(X.flatten(), f, alpha=0.3)

ax2.set_title("Samples from Gaussianized IP using Moments")

w = f1.get_weights()

J = jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X), w)

J = torch.cat([torch.flatten(a, -2, -1) for a in J], dim=-1).transpose(-1, -2)
S = torch.exp(f1.get_std_params()) ** 2
cov = torch.einsum("nsd, s, msd -> nmd", J, S, J).squeeze(-1)

pos = ax6.imshow(cov.detach().numpy())
fig.colorbar(pos, ax=ax6)
ax6.set_title("Covariance matrix Taylor (1)")

mean = f1.forward_mean(torch.tensor(X, dtype=args.dtype)).squeeze(-1).detach().numpy()

print(mean[0])
print(mean[-1])
L = np.linalg.cholesky(cov.detach().numpy() + np.eye(mean.shape[0]) * 1e-6)
z = np.random.randn(20, mean.shape[0])


samples = mean + (L @ z.T).T

ax3.plot(X.flatten(), mean)
ax3.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov.detach().numpy())),
    alpha=0.2,
)
for f in samples:
    ax3.plot(X.flatten(), f, alpha=0.3)
ax3.set_title("Samples from Gaussianized IP using Taylor (1)")


GP_mean = f1.GP_mean(torch.tensor(X)).detach().numpy()

GP_cov = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    GP_cov[i] = f1.GP_cov(torch.tensor(X[i]).unsqueeze(0), torch.tensor(X[i]).unsqueeze(0)).detach().numpy()
    
    
ax4.plot(test_dataset.inputs.flatten(), GP_mean, alpha=1, label = "GP Mean")
ax4.set_title("GP distribution")
ax4.fill_between(
    test_dataset.inputs.flatten(),
    GP_mean.flatten() - 2*GP_cov,
    GP_mean.flatten() + 2*GP_cov,
    alpha = 0.3,
)


l = list(itertools.combinations_with_replacement(range(X.shape[0]), 2))
GP_cov = np.zeros((X.shape[0], X.shape[0]))
for a in l:
    print(a)
    GP_cov[a[0], a[1]] = f1.GP_cov(torch.tensor(X[a[0]]).unsqueeze(0), torch.tensor(X[a[1]]).unsqueeze(0)).detach().numpy()
    GP_cov[a[1], a[0]] = GP_cov[a[0], a[1]]
pos = ax7.imshow(GP_cov)
ax7.set_title("Covariance matrix (GP")
fig.colorbar(pos, ax=ax7)

L = np.linalg.cholesky(GP_cov + np.eye(GP_mean.shape[0]) * 1e-5)
z = np.random.randn(20, GP_mean.shape[0])


samples = GP_mean.T + (L @ z.T).T

for f in samples:
    ax4.plot(X.flatten(), f, alpha=0.3)


# xmean = f1.get_weights()

# shapes = [list(a.shape) for a in xmean]
# sizes = [np.prod(a) for a in shapes]

# xmean = torch.cat([torch.flatten(a, -2, -1) for a in xmean], dim=-1).detach().numpy()
# xcov = torch.exp(f1.get_std_params()).detach().numpy() ** 2

# L = xmean.shape[0]
# alpha = 1e-3
# beta = 2

# lamb = alpha ** 2 * L - L


# sqrt = np.sqrt((L + lamb) * xcov)
# barx = [xmean]
# for i in range(L):
#     a = np.copy(xmean)
#     a[i] += sqrt[i]
#     barx.append(a)

# for i in range(L, 2 * L):
#     a = np.copy(xmean)
#     a[i - L] -= sqrt[i - L]
#     barx.append(a)


# Wm = [torch.tensor(lamb / (L + lamb))]
# Wc = [torch.tensor(lamb / (L + lamb) + (1 - alpha ** 2 + beta))]
# for i in range(1, 2 * L + 1):
#     Wm.append(torch.tensor(1 / (2 * (L + lamb))))
#     Wc.append(torch.tensor(1 / (2 * (L + lamb))))


# Y = []
# for x in barx:

#     v = []
#     for i in range(len(shapes)):
#         aux = x[int(np.sum(sizes[:i])) : int(np.sum(sizes[: i + 1]))]
#         aux = aux.reshape(shapes[i])
#         v.append(torch.tensor(aux, dtype=args.dtype))

#     Y.append(f1.forward_weights(torch.tensor(X, dtype=args.dtype), v).unsqueeze(0))

# Y = torch.concat(Y, dim=0)
# Wm = torch.stack(Wm).unsqueeze(-1).unsqueeze(-1)
# Wc = torch.stack(Wc)

# ymean = torch.sum(Y * Wm, 0)
# a = (Y - ymean).squeeze(-1)
# Ycov = torch.einsum("s,sn,sm->nm", Wc, a, a).detach().numpy()


# ax7.imshow(Ycov)
# ax7.set_title("Covariance matrix Kalman")


# ax4.plot(X.flatten(), ymean.flatten())
# ax4.fill_between(
#     X.flatten(),
#     ymean.flatten() - 2 * np.sqrt(np.diag(Ycov)),
#     ymean.flatten() + 2 * np.sqrt(np.diag(Ycov)),
#     alpha=0.2,
# )

# L = np.linalg.cholesky(Ycov + np.eye(ymean.shape[0]) * 1e-4)
# z = np.random.randn(20, ymean.shape[0])


# samples = ymean.flatten() + (L @ z.T).T


# for f in samples:
#     ax4.plot(X.flatten(), f, alpha=0.3)
# ax4.set_title("Samples from Gaussianized IP using Kalman")

# plt.tight_layout()
# plt.savefig("figure.pdf")
plt.show()
