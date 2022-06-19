from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, IVAE, TVIP
from src.layers_init import init_layers
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args.device)
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

args.likelihood = QuadratureGaussian()

# Get VIP layers
layers = init_layers(train_dataset.inputs, args.dataset.output_dim, **vars(args))

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Create DVIP object
dvip = TVIP(
    args.likelihood,
    layers[0],
    len(train_dataset),
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

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
# ax5 = plt.subplot(3, 1, 3)

x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
ax1.scatter(train_dataset.inputs, y, label="Training", s=3)
ylims = ax1.get_ylim()

F, Fmean, Fvar = dvip(torch.tensor(train_dataset.inputs, device=args.device), 100)
print(F.shape)

F = F.squeeze().cpu().detach().numpy()
Fmean = Fmean.squeeze().cpu().detach().numpy()
Fvar = Fvar.squeeze().cpu().detach().numpy()
diag_var = np.sqrt(np.diag(Fvar))

sort = np.argsort(train_dataset.inputs.flatten())
ax3.plot(train_dataset.inputs.flatten()[sort], F.T[sort][:, 1:], alpha=0.5)
ax3.plot(
    train_dataset.inputs.flatten()[sort],
    F.T[sort][:, 0],
    label=r"$\mathbf{f}_K$ posterior samples",
)
ax2.plot(
    train_dataset.inputs.flatten()[sort],
    Fmean[sort],
    color="teal",
    label=r"$\mathbf{f}_0$ mean",
)
ax2.fill_between(
    train_dataset.inputs.flatten()[sort],
    Fmean[sort] - 2 * diag_var[sort],
    Fmean[sort] + 2 * diag_var[sort],
    alpha=0.1,
    color="teal",
    label=r"$\mathbf{f}_0$ std",
)

# ax3.plot(
#     np.arange(len(loss) // 3, len(loss)), loss[len(loss) // 3 :], label="Loss function"
# )

x = np.tile(x.flatten(), F.shape[0])
ax4.hist2d(x, F.flatten(), bins=100)

ax1.legend()
ax2.legend()
ax3.legend()
# ax5.legend()
plt.tight_layout()
plt.savefig("fig_10.pdf")
plt.show()
