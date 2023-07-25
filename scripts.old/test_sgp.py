from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer


from scipy.cluster.vq import kmeans2

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import SparseLA
from src.likelihood import Gaussian
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

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
    args.seed,
    torch.nn.Tanh,
    args.device,
    args.dtype,
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

Z = kmeans2(train_dataset.inputs, 20, minit="points", seed=args.seed)[0]

sparseLA = SparseLA(
    f.forward,
    Z,
    alpha=args.bb_alpha,
    prior_variance=2.2026465**2,
    likelihood=Gaussian(
        log_variance=np.log(0.11802231**2), device=args.device, dtype=args.dtype
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


print("Last NELBO value: ", loss[-1])

import matplotlib.pyplot as plt

fig, axis = plt.subplots(2, 1)
a = np.arange(len(loss) // 3, len(loss))
axis[0].plot(a, loss[len(loss) // 3 :])

axis[1].plot(loss)

axis[1].set_yscale("symlog")
plt.show()


import matplotlib.pyplot as plt

fig, axis = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]}, figsize=(16, 10))


axis[0].scatter(
    train_dataset.inputs,
    train_dataset.targets,
    label="Training points",
    color="black",
    alpha=0.9,
)


X = np.concatenate([train_dataset.inputs, test_dataset.inputs], 0)
Z = sparseLA.inducing_locations.detach().cpu().numpy()


f_mu, f_std = sparseLA(torch.tensor(X, dtype=torch.float64, device=args.device))
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_std = f_std.squeeze().detach().cpu().numpy()
pred_std = np.sqrt(
    f_std**2 + (torch.exp(sparseLA.likelihood.log_variance)).detach().cpu().numpy()
)


sort = np.argsort(X.flatten())


axis[0].plot(X.flatten()[sort], f_mu.flatten()[sort], label="Predictions")
axis[0].fill_between(
    X.flatten()[sort],
    f_mu.flatten()[sort] - 2 * pred_std[sort],
    f_mu.flatten()[sort] + 2 * pred_std[sort],
    alpha=0.2,
    label="SparseLA uncertainty",
)

m = f(sparseLA.inducing_locations).flatten().detach().cpu().numpy()

xlims = axis[0].get_xlim()

axis[0].scatter(
    sparseLA.inducing_locations.detach().cpu().numpy(), m, label="Inducing locations"
)

axis[1].fill_between(
    X.flatten()[sort],
    np.zeros(X.shape[0]),
    pred_std[sort],
    alpha=0.2,
    label="SparseLA uncertainty (std)",
)
axis[1].fill_between(
    X.flatten()[sort],
    np.zeros(X.shape[0]),
    torch.sqrt(torch.exp(sparseLA.likelihood.log_variance)).detach().cpu().numpy(),
    alpha=0.2,
    label="Likelihood uncertainty (std)",
)

axis[0].set_xlim(left=xlims[0], right=xlims[1])

axis[0].xaxis.set_tick_params(labelsize=20)
axis[0].yaxis.set_tick_params(labelsize=20)
axis[1].xaxis.set_tick_params(labelsize=20)
axis[1].yaxis.set_tick_params(labelsize=20)
axis[0].legend(prop={"size": 14}, loc="upper left")
axis[1].legend(prop={"size": 14}, loc="upper left")

axis[0].set_ylim(-3, 3)
plt.savefig("SparseLA M={}.pdf".format(Z.shape[0]), format="pdf")

plt.show()
