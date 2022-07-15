from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, IVAE, TVIP2
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

train_d, train_test_d, test_d = args.dataset.get_split(args.test_size, args.split)
train_loader = DataLoader(train_d, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_d, batch_size=args.batch_size)
test_loader = DataLoader(test_d, batch_size=args.batch_size)


layers = init_layers(train_d.inputs, args.dataset.output_dim, **vars(args))
train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

dvip = DVIP_Base(
    args.likelihood,
    layers,
    len(train_d),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_d.targets_mean,
    y_std=train_d.targets_std,
    dtype=args.dtype,
    device=args.device,
)
dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


dvip.num_samples = args.num_samples_train
# Train the model
losses = fit(
    dvip,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)

dvip.print_variables()


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


def get_predictive_results(mean, var):

    prediction_mean = np.mean(mean, axis=0)
    prediction_var = np.mean(var + mean ** 2, axis=0) - prediction_mean ** 2
    return prediction_mean, prediction_var


test_mean, test_std = predict(dvip, test_loader, device=args.device)
test_prediction_mean, test_prediction_var = get_predictive_results(
    test_mean, test_std ** 2
)

sort = np.argsort(test_dataset.inputs.flatten())
ax2.plot(
    test_dataset.inputs.flatten()[sort],
    test_prediction_mean[sort],
    color="orange",
    label="Predictive mean",
)
ax2.fill_between(
    test_dataset.inputs.flatten()[sort],
    (test_prediction_mean - 2 * np.sqrt(test_prediction_var)).flatten()[sort],
    (test_prediction_mean + 2 * np.sqrt(test_prediction_var)).flatten()[sort],
    color="orange",
    alpha=0.3,
    label="Predictive std",
)

F = dvip.get_prior_samples(torch.tensor(test_dataset.inputs, device=args.device))

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
