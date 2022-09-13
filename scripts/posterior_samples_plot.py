from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, TVIP
from src.layers_init import init_layers
from src.layers import TVIPLayer, TVIP3Layer, TVIP2Layer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args.device)
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Get VIP layers
f = BayesianNN(
    num_samples=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    structure=args.bnn_structure,
    activation=args.activation,
    output_dim=train_dataset.output_dim,
    layer_model=args.bnn_layer,
    dropout=args.dropout,
    fix_random_noise=args.fix_prior_noise,
    zero_mean_prior=args.zero_mean_prior,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

layer = TVIP2Layer(
    f,
    num_regression_coeffs=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    output_dim=train_dataset.output_dim,
    add_prior_regularization=args.prior_kl,
    mean_function=None,
    log_layer_noise=None,
    n_coupling=args.n_coupling,
    dtype=args.dtype,
    device=args.device,
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Create DVIP object
dvip = TVIP(
    likelihood=args.likelihood,
    layer=layer,
    num_data=len(train_dataset),
    num_samples=args.num_samples_train,
    bb_alpha=args.bb_alpha,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
if args.freeze_prior:
    dvip.freeze_prior()

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

x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
ax1.scatter(train_dataset.inputs, y, label="Training dataset", s=5)
x = test_dataset.inputs
y = test_dataset.targets
ax1.scatter(test_dataset.inputs, y, label="Test dataset", s=5)
ax1.set_title("Dataset")
ylims = ax1.get_ylim()

X = torch.tensor(test_dataset.inputs, device=args.device)
F, std = dvip(X, 100)
print(F.shape)

F = F.squeeze().cpu().detach().numpy()
std = std.squeeze().cpu().detach().numpy()
sort = np.argsort(test_dataset.inputs.flatten())


ax2.plot(test_dataset.inputs.flatten()[sort], F.T[sort], alpha=0.5)

ax2.set_title(r"$Q(\mathbf{f})$ samples")

#     ax2.plot
# ax2.plot(test_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
# ax2.plot(
#     test_dataset.inputs.flatten()[sort],
#     F.T[sort][:, 0],
#     label=r"$Q(\mathbf{f})$ samples",
# )

Fprior, F = dvip.get_prior_samples(X, 100)

F = F.squeeze().cpu().detach().numpy()
ax3.plot(test_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
ax3.plot(
    test_dataset.inputs.flatten()[sort],
    F.T[sort][:, 0],
)
ax3.set_title(
    "Prior BNN Samples",
)

Fprior = Fprior.squeeze().cpu().detach().numpy()
ax4.plot(test_dataset.inputs.flatten()[sort], Fprior.T[sort], alpha=0.5)
ax4.set_title(
    r"$P(\mathbf{f})$ samples",
)

plt.tight_layout()
plt.savefig("plots/tvip_" + create_file_name(args) + ".pdf", format="pdf")
plt.show()


n = 20
fig, axes = plt.subplots(n // 4, 4, figsize=(12, 6))
a0, ak = dvip.layer.get_samples(1000)
for i in range(n):
    axes[i // 4][i % 4].hist(
        a0.squeeze().cpu().detach().numpy()[:, i],
        bins=50,
        density=True,
        alpha=0.5,
        label=r"$Q(a_0)$",
    )
    axes[i // 4][i % 4].hist(
        ak.squeeze().cpu().detach().numpy()[:, i],
        bins=50,
        density=True,
        alpha=0.5,
        label=r"$Q(a_k)$",
    )
    axes[i // 4][i % 4].set_title("Distribution of Coefficient {}".format(i))
    axes[i // 4][i % 4].legend()
plt.tight_layout()
plt.savefig("plots/flow_" + create_file_name(args) + ".pdf", format="pdf")
plt.show()
