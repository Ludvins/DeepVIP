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
from src.sparseLA import GPLLA, ELLA, SparseLA
from src.likelihood import Bernoulli
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset, Test_Dataset
from torchsummary import summary

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
    torch.nn.Sigmoid,
    args.device,
    args.dtype,
)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.BCELoss()

# Set the number of training samples to generate
# Train the model
start = timer()

loss = fit_map(
    f,
    train_loader,
    opt,
    criterion=criterion,
    use_tqdm=True,
    return_loss=True,
    iterations=args.MAP_iterations,
    device=args.device,
)
end = timer()


lla = GPLLA(
    f[:-1],
    prior_std=1.0,
    likelihood_hessian=lambda x, y: (f(x) * (1 - f(x))).unsqueeze(-1).permute(1, 2, 0),
    likelihood=Bernoulli(device=args.device, dtype=args.dtype),
    backend=BackPackInterface(f, "classification"),
    device=args.device,
    dtype=args.dtype,
)


lla.fit(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
)


import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams["pdf.fonttype"] = 42
fig, axis = plt.subplots(4, 2, figsize=(15, 20))


axis[0][0].scatter(
    train_dataset.inputs, train_dataset.targets, alpha=0.8, label="Training Dataset"
)
axis[0][0].set_title("Training Dataset")


def plot_model(model, index, name):
    mean, var = predict(model, test_loader)
    p = mean
    std = np.sqrt(var)

    sort = np.argsort(test_dataset.inputs.flatten())

    axis[index][0].plot(
        test_dataset.inputs.flatten()[sort],
        mean.flatten()[sort],
        label="Predictive mean",
        color="teal",
    )

    mean, var = model(
        torch.tensor(test_dataset.inputs, device=args.device, dtype=args.dtype)
    )

    mean = mean.detach().cpu().numpy()
    std = np.sqrt(var.detach().cpu().numpy())

    sort = np.argsort(test_dataset.inputs.flatten())

    axis[index][1].plot(
        test_dataset.inputs.flatten()[sort],
        mean.flatten()[sort],
        label="Predictive mean",
        color="teal",
    )
    axis[index][1].fill_between(
        test_dataset.inputs.flatten()[sort],
        mean.flatten()[sort] - 2 * std.flatten()[sort],
        mean.flatten()[sort] + 2 * std.flatten()[sort],
        label="Predictive std",
        alpha=0.1,
        color="teal",
    )

    axis[index][0].set_title("Predictive probability of {}".format(name))
    axis[index][1].set_title("Latent Distribution of {}".format(name))

    return mean, std


mean_lla, std_lla = plot_model(lla, 1, "LLA")

ella = ELLA(
    f[:-1],
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=1.0,
    likelihood_hessian=lambda x, y: (f(x) * (1 - f(x))).unsqueeze(-1).permute(1, 2, 0),
    likelihood=Bernoulli(device=args.device, dtype=args.dtype),
    backend=BackPackInterface(f, "classification"),
    seed=args.seed,
    device=args.device,
    dtype=args.dtype,
)


ella.fit(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
)

mean_ella, std_ella = plot_model(ella, 2, "ELLA")

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

sparseLA = SparseLA(
    f[:-1].forward,
    Z,
    alpha=args.bb_alpha,
    prior_std=1,
    likelihood=Bernoulli(device=args.device, dtype=args.dtype),
    num_data=train_dataset.inputs.shape[0],
    output_dim=1,
    backend=BackPackInterface(f, "classification"),
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
mean_valla, std_valla = plot_model(sparseLA, 3, "VaLLA")

KL_ELLA = (
    -np.log(std_lla) + np.log(std_ella) - 0.5 + ((std_lla**2) / (2 * (std_ella**2)))
)
KL_ELLA = np.sum(KL_ELLA)
KL_VaLLA = (
    -np.log(std_lla)
    + np.log(std_valla)
    - 0.5
    + ((std_lla**2) / (2 * (std_valla**2)))
)
KL_VaLLA = np.sum(KL_VaLLA)

print("MAE ELLA ", np.mean(np.abs(std_lla - std_ella)))
print("MAE VaLLA ", np.mean(np.abs(std_lla - std_valla)))

print("KL ELLA: ", KL_ELLA)
print("KL VaLLA: ", KL_VaLLA)


plt.show()
