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

X = test_dataset.inputs
print(X.shape)
f1 = GP(
    input_dim=X.shape[1],
    output_dim=1,
    inner_layer_dim=20,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)



import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
fig = plt.gcf()

samples = f1(torch.tensor(X, dtype=args.dtype), 50000)
for f in samples[:100]:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")

mean = torch.mean(samples, dim=0).squeeze().detach().numpy()
cov = torch.cov(samples.squeeze().T)

pos = ax2.imshow(cov.detach().numpy())

fig.colorbar(pos, ax=ax2)
ax2.set_title("Covariance matrix (Empirical)")

ax3.plot(X.flatten(), mean, label = "Empirical mean")
ax3.fill_between(
    X.flatten(),
    mean - 2 * np.sqrt(np.diag(cov.detach().numpy())),
    mean + 2 * np.sqrt(np.diag(cov.detach().numpy())),
    alpha=0.2,
    label = "Empirical std"
)

ax3.set_title("Samples from Gaussianized IP using Moments")

GP_mean = f1.GP_mean(torch.tensor(X)).detach().numpy()
GP_cov = f1.GP_cov(torch.tensor(X))

    
ax3.plot(test_dataset.inputs.flatten(), GP_mean, alpha=1, label = "GP Mean")
ax3.set_title("GP distribution")


# l = list(itertools.combinations_with_replacement(range(X.shape[0]), 2))
# GP_cov = np.zeros((X.shape[0], X.shape[0]))
# for a in l:
#     print(a)
#     GP_cov[a[0], a[1]] = f1.GP_cov(torch.tensor(X[a[0]]).unsqueeze(0), torch.tensor(X[a[1]]).unsqueeze(0)).detach().numpy()
#     GP_cov[a[1], a[0]] = GP_cov[a[0], a[1]]



ax3.fill_between(
    test_dataset.inputs.flatten(),
    GP_mean.flatten() - 2*np.sqrt(torch.diag(GP_cov).detach().numpy()),
    GP_mean.flatten() + 2*np.sqrt(torch.diag(GP_cov).detach().numpy()),
    alpha = 0.3,
    label = "GP std",
)

pos = ax4.imshow(GP_cov.detach().numpy())
ax4.set_title("Covariance matrix (GP)")
fig.colorbar(pos, ax=ax4)

# L = np.linalg.cholesky(GP_cov + np.eye(GP_mean.shape[0]) * 1e-5)
# z = np.random.randn(20, GP_mean.shape[0])


# samples = GP_mean.T + (L @ z.T).T

# for f in samples:
#     ax4.plot(X.flatten(), f, alpha=0.3)

ax3.legend()
plt.tight_layout()
plt.savefig("GP_cov.pdf", dpi=1200, format="pdf")
plt.show()
