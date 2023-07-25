from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

from scipy.cluster.vq import kmeans2

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit, predict
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import SparseLA, GPLLA, OptimalSparseLA
from src.likelihood import Gaussian
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset


args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

args.dataset = get_dataset(args.dataset_name)
train_dataset, full_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
full_loader = DataLoader(full_dataset, batch_size=args.batch_size)
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
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
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

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]


prior_std = 2.2026465
ll_std = 0.11802231


sparseLA = OptimalSparseLA(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    f.forward,
    Z,
    alpha=args.bb_alpha,
    prior_std=prior_std,
    likelihood=Gaussian(
        device=args.device, log_variance=np.log(ll_std**2), dtype=args.dtype
    ),
    num_data=train_dataset.inputs.shape[0],
    output_dim=1,
    backend=BackPackInterface(f, "regression"),
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,
)

if args.freeze_ll:
    sparseLA.freeze_ll()


sparseLA.print_variables()

opt = torch.optim.Adam(sparseLA.parameters(), lr=args.lr)

start = timer()
loss = fit(
    sparseLA,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

sparseLA.print_variables()


lla = GPLLA(
    f,
    prior_std=prior_std,
    likelihood_hessian=lambda x, y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0)
    / ll_std**2,
    likelihood=Gaussian(
        device=args.device, log_variance=np.log(ll_std**2), dtype=args.dtype
    ),
    backend=BackPackInterface(f, "regression"),
    device=args.device,
    dtype=args.dtype,
)


lla.fit(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
)


import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
fig, axis = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]}, figsize=(16, 10))


axis[0].scatter(train_dataset.inputs, train_dataset.targets, label="Training points")


Z = sparseLA.inducing_locations.detach().cpu().numpy()


valla_mean, valla_var = predict(sparseLA, test_loader)
valla_std = np.sqrt(valla_var).flatten()

_, valla_prior_var = sparseLA.forward_prior(
    torch.tensor(test_dataset.inputs, device=args.device, dtype=args.dtype)
)
valla_prior_std = np.sqrt(valla_prior_var.detach().cpu().numpy().flatten())

sort = np.argsort(test_dataset.inputs.flatten())


axis[0].plot(
    test_dataset.inputs.flatten()[sort],
    valla_mean.flatten()[sort],
    label="Predictions",
    color="black",
)
axis[0].fill_between(
    test_dataset.inputs.flatten()[sort],
    valla_mean.flatten()[sort] - 2 * valla_std[sort],
    valla_mean.flatten()[sort] + 2 * valla_std[sort],
    alpha=0.2,
    label="VaLLA uncertainty",
    color="orange",
)

axis[0].fill_between(
    test_dataset.inputs.flatten()[sort],
    valla_mean.flatten()[sort] - 2 * (valla_prior_std[sort] + ll_std),
    valla_mean.flatten()[sort] + 2 * (valla_prior_std[sort] + ll_std),
    alpha=0.2,
    label="Prior uncertainty",
)

m = f(sparseLA.inducing_locations).flatten().detach().cpu().numpy()

xlims = axis[0].get_xlim()

axis[0].scatter(
    sparseLA.inducing_locations.detach().cpu().numpy(),
    m,
    label="Inducing locations",
    color="darkorange",
)

axis[1].fill_between(
    test_dataset.inputs.flatten()[sort],
    np.zeros(test_dataset.inputs.shape[0]),
    valla_std[sort],
    alpha=0.2,
    label="VaLLA uncertainty (std)",
    color="orange",
)

axis[0].set_xlim(left=xlims[0], right=xlims[1])

inducing_history = np.stack(sparseLA.inducing_history)
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, inducing_history.shape[1]))

axis[0].legend()
axis[1].legend()

axis[0].set_title("Predictive distribution")
axis[1].set_title("Uncertainty decomposition")


lla_mean, lla_var = predict(lla, test_loader)

lla_std = np.sqrt(lla_var).flatten()


sort = np.argsort(test_dataset.inputs.flatten())
axis[0].plot(
    test_dataset.inputs.flatten()[sort],
    lla_mean.flatten()[sort],
    alpha=0.2,
    color="teal",
)
axis[0].fill_between(
    test_dataset.inputs.flatten()[sort],
    lla_mean.flatten()[sort] - 2 * lla_std[sort],
    lla_mean.flatten()[sort] + 2 * lla_std[sort],
    alpha=0.2,
    label="GP uncertainty",
    color="teal",
)


axis[1].fill_between(
    test_dataset.inputs.flatten()[sort],
    np.zeros(test_dataset.inputs.shape[0]),
    lla_std[sort],
    alpha=0.2,
    label="GP uncertainty (std)",
    color="teal",
)


axis[0].xaxis.set_tick_params(labelsize=20)
axis[0].yaxis.set_tick_params(labelsize=20)
axis[1].xaxis.set_tick_params(labelsize=20)
axis[1].yaxis.set_tick_params(labelsize=20)

axis[0].legend(prop={"size": 14}, loc="upper left")
axis[1].legend(prop={"size": 14}, loc="upper left")

plt.show()
