from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2

sys.path.append(".")

from src.fvi import Test, Test2
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

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

Z = kmeans2(train_dataset.inputs, 5, minit='points', seed = args.seed)[0]

f1 = BayesianNN(
                num_samples=20,
                input_dim=1,
                structure=args.bnn_structure,
                activation=args.activation,
                output_dim=1,
                layer_model=args.bnn_layer,
                dropout=args.dropout,
                fix_random_noise=False,
                zero_mean_prior=args.zero_mean_prior,
                device=args.device,
                seed=args.seed,
                dtype=args.dtype,
            )

f1.freeze_parameters()

f2 = BayesianNN(
                num_samples=20,
                input_dim=1,
                structure=args.bnn_structure,
                activation=args.activation,
                output_dim=1,
                layer_model=args.bnn_layer,
                dropout=args.dropout,
                fix_random_noise=False,
                zero_mean_prior=args.zero_mean_prior,
                device=args.device,
                seed=args.seed,
                dtype=args.dtype,
            )
# Create DVIP object
dvip = Test(
    prior_ip=f1,
    variational_ip=f2,
    Z = Z,
    likelihood=args.likelihood,
    num_data=len(train_dataset),
    num_samples=args.num_samples_train,
    bb_alpha=args.bb_alpha,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
#dvip.freeze_ll_variance()
dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

# Set the number of training samples to generate
dvip.num_samples = args.num_samples_train
# Train the model
start = timer()
loss = fit(
    dvip,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

n = len(loss)
ax1.plot(np.arange(n // 5, n), loss[n // 5 :])
ax2.plot(np.arange(n // 5, n), dvip.bb_alphas[n // 5 :])
ax3.plot(np.arange(n // 5, n), dvip.KLs[n // 5 :])
ax4.plot(loss)
ax5.plot(dvip.bb_alphas)
ax6.plot(dvip.KLs)
ax4.set_yscale("symlog")
ax5.set_yscale("symlog")
ax6.set_yscale("symlog")
ax1.set_title("Loss")
ax2.set_title("Data Fitting Term")
ax3.set_title("Regularizer Term")

plt.savefig("plots/loss_" + create_file_name(args) + ".pdf", format="pdf")
plt.show()
dvip.print_variables()

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
# ax5 = plt.subplot(3, 1, 3)


x = test_dataset.inputs
y = test_dataset.targets
if y is not None:
    ax1.scatter(test_dataset.inputs, y, label="Test dataset", s=5)

x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
ax1.scatter(train_dataset.inputs, y, label="Training dataset", s=5)

ax1.set_title("Dataset")
ylims = ax1.get_ylim()

X = torch.tensor(test_dataset.inputs, device=args.device)
F, std = dvip(X, 100)
print(F.shape)
print(std.shape)
u = dvip.generate_u_samples().detach().numpy() * train_dataset.targets_std + train_dataset.targets_mean

F = F.squeeze(-1).cpu().detach().numpy()
std = std.squeeze(-1).cpu().detach().numpy()
sort = np.argsort(test_dataset.inputs.flatten())


for i in range(F.shape[0]):
    #ax2.plot(test_dataset.inputs.flatten()[sort], F[i, sort], alpha=0.5)
    ax2.fill_between(test_dataset.inputs.flatten()[sort],
                     F[i, sort] - 3*std[i, sort],
                     F[i, sort] + 3*std[i, sort],
                     alpha=0.1)


ax2.set_title(r"$Q(\mathbf{f})$ samples")

ax2.plot(test_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
ax2.plot(
    test_dataset.inputs.flatten()[sort],
    F.T[sort][:, 0],
    label=r"$Q(\mathbf{f})$ samples",
)
ax2.scatter(train_dataset.inputs, y, s=5, zorder = 2)

ax3.set_title(r"$Q(\mathbf{u})$ samples")

for i in range(u.shape[0]):
    ax3.scatter(dvip.inducing_points.detach().numpy(), u[i], alpha=0.5)
ax3.scatter(Z, np.zeros_like(Z), alpha=0.9, color = "black")

Fprior = dvip.get_prior_samples(X, 100)

Fprior = Fprior.squeeze().cpu().detach().numpy()
ax4.plot(test_dataset.inputs.flatten()[sort], Fprior.T[sort], alpha=0.5)
ax4.set_title(
    r"$P(\mathbf{f})$ samples",
)

plt.tight_layout()
plt.savefig("plots/tvip_" + create_file_name(args) + ".pdf", format="pdf")
plt.show()
