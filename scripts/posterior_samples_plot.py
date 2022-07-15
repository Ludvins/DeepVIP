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
from src.layers import TVIPLayer
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

layer = TVIPLayer(
    f,
    num_regression_coeffs=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    output_dim=train_dataset.output_dim,
    add_prior_regularization=args.prior_kl,
    mean_function=None,
    q_mu_initial_value=0,
    log_layer_noise=-5,
    q_sqrt_initial_value=1,
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
    bb_alpha=args.bb_alpha,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
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


dvip.print_variables()

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
# ax5 = plt.subplot(3, 1, 3)

x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
ax1.scatter(train_dataset.inputs, y, label="Training dataset", s=5)
x = test_dataset.inputs
y = test_dataset.targets
ax1.scatter(test_dataset.inputs, y, label="Test dataset", s=5)

ylims = ax1.get_ylim()

X = torch.tensor(test_dataset.inputs, device=args.device)
F, std = dvip.predict_y(X, 100)


F = F.squeeze().cpu().detach().numpy()
std = std.squeeze().cpu().detach().numpy()
sort = np.argsort(test_dataset.inputs.flatten())

for f, s in zip(F, std):
    ax2.plot(
        test_dataset.inputs.flatten()[sort],
        f[sort],
    )
    ax2.fill_between(
        test_dataset.inputs.flatten()[sort],
        f[sort] - 2 * s[sort],
        f[sort] + 2 * s[sort],
        alpha=0.1,
    )

#     ax2.plot
# ax2.plot(test_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
# ax2.plot(
#     test_dataset.inputs.flatten()[sort],
#     F.T[sort][:, 0],
#     label=r"$Q(\mathbf{f})$ samples",
# )

F = dvip.get_prior_samples(X)

F = F.squeeze().cpu().detach().numpy()
ax3.plot(test_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
ax3.plot(
    test_dataset.inputs.flatten()[sort],
    F.T[sort][:, 0],
    label=r"$P(\mathbf{f})$ samples",
)

ax1.legend()
ax3.legend()
ax2.legend()
plt.tight_layout()
plt.savefig("flow_on_a2.pdf")
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
plt.savefig("flow_on_a2_.pdf")
plt.show()
