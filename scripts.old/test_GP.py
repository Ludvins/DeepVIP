from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

from laplace.curvature import AsdlInterface, BackPackInterface
from laplace import Laplace

from scipy.cluster.vq import kmeans2

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit, predict
from scripts.filename import create_file_name
from src.generative_functions import *
from src.likelihood import Gaussian
from utils.dataset import get_dataset
from src.sparseLA import GPLLA
from utils.models import get_mlp

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)
args.dataset = get_dataset(args.dataset_name)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = get_mlp(
    train_dataset.inputs.shape[1],
    train_dataset.targets.shape[1],
    [50, 50],
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()

# Set the number of training samples to generate
# Train the model
start = timer()

loss = fit_map(
    f,
    train_loader,
    opt,
    criterion=torch.nn.MSELoss(),
    use_tqdm=True,
    return_loss=True,
    iterations=args.MAP_iterations,
    device=args.device,
)
end = timer()

lla = GPLLA(
    f,
    prior_std=2.183941,
    likelihood_hessian=lambda x, y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0)
    / 0.11669471**2,
    likelihood=Gaussian(
        device=args.device, log_variance=np.log(0.11669471**2), dtype=args.dtype
    ),
    backend=BackPackInterface(f, "classification"),
    device=args.device,
    dtype=args.dtype,
)


lla.fit(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
)

import matplotlib.pyplot as plt


fig, axis = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})

axis[0].scatter(train_dataset.inputs, train_dataset.targets, label="Training points")
# plt.scatter(test_dataset.inputs, test_dataset.targets, label = "Test points")


lla_mean, lla_var = predict(lla, test_loader)
lla_std = np.sqrt(lla_var).flatten()

sort = np.argsort(test_dataset.inputs.flatten())

axis[0].plot(
    test_dataset.inputs.flatten()[sort], lla_mean.flatten()[sort], label="Predictions"
)
axis[0].fill_between(
    test_dataset.inputs.flatten()[sort],
    lla_mean.flatten()[sort] - 2 * lla_std[sort],
    lla_mean.flatten()[sort] + 2 * lla_std[sort],
    alpha=0.1,
    label="GP_LA uncertainty",
)


axis[0].legend()

axis[1].fill_between(
    test_dataset.inputs.flatten()[sort],
    np.zeros(test_dataset.inputs.shape[0]),
    lla_std[sort],
    alpha=0.1,
    label="GP uncertainty (std)",
)

axis[1].legend()
plt.show()
