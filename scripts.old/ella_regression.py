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
from utils.pytorch_learning import fit_map, fit, forward, score
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import ELLA, GPLLA
from src.likelihood import Gaussian
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset
from utils.metrics import MetricsRegression

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

if args.weight_decay != 0:
    args.prior_std = np.sqrt(1 / (len(train_dataset) * args.weight_decay))


ll_var = 0.1

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()

# Set the number of training samples to generate

try:
    f.load_state_dict(torch.load("weights/regression_weights_" + args.dataset_name))
except:
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

    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/regression_weights_" + args.dataset_name)

import numpy

numpy.set_printoptions(threshold=sys.maxsize)


ella = ELLA(
    f,
    f.output_size,
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=args.prior_std,
    likelihood_hessian=lambda x, y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0)
    / ll_var,
    likelihood=Gaussian(
        device=args.device, log_variance=np.log(ll_var), dtype=args.dtype
    ),
    backend=BackPackInterface(f, f.output_size),
    seed=args.seed,
    device=args.device,
    dtype=args.dtype,
)


ella.fit(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
)


lla = GPLLA(
    f,
    prior_std=args.prior_std,
    likelihood_hessian=lambda x, y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0)
    / ll_var,
    likelihood=Gaussian(
        device=args.device, log_variance=np.log(ll_var), dtype=args.dtype
    ),
    backend=BackPackInterface(f, f.output_size),
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
""" if test_dataset.targets is not None:
    axis[0].scatter(test_dataset.inputs, test_dataset.targets, label = "Test points") """

ella_mean, ella_var = forward(ella, full_loader)
ella_pred_var = ella_var + ll_var
ella_std = np.sqrt(ella_var).flatten()
ella_pred_std = np.sqrt(ella_pred_var).flatten()


sort = np.argsort(full_dataset.inputs.flatten())


axis[0].plot(
    full_dataset.inputs.flatten()[sort],
    ella_mean.flatten()[sort],
    label="Predictions",
    color="black",
)
axis[0].fill_between(
    full_dataset.inputs.flatten()[sort],
    ella_mean.flatten()[sort] - 2 * ella_pred_std[sort],
    ella_mean.flatten()[sort] + 2 * ella_pred_std[sort],
    alpha=0.2,
    label="ELLA uncertainty",
    color="orange",
)


xlims = axis[0].get_xlim()

axis[1].fill_between(
    full_dataset.inputs.flatten()[sort],
    np.zeros(full_dataset.inputs.shape[0]),
    ella_pred_std[sort],
    alpha=0.2,
    label="ELLA uncertainty (std)",
    color="orange",
)

axis[0].set_xlim(left=xlims[0], right=xlims[1])


axis[0].legend()
axis[1].legend()

axis[0].set_title("Predictive distribution")
axis[1].set_title("Uncertainty decomposition")


lla_mean, lla_var = forward(lla, full_loader)
lla_pred_var = lla_var + ll_var
lla_pred_std = np.sqrt(lla_pred_var).flatten()
lla_std = np.sqrt(lla_var).flatten()


sort = np.argsort(full_dataset.inputs.flatten())
axis[0].plot(
    full_dataset.inputs.flatten()[sort],
    lla_mean.flatten()[sort],
    alpha=0.2,
    color="teal",
)
axis[0].fill_between(
    full_dataset.inputs.flatten()[sort],
    lla_mean.flatten()[sort] - 2 * lla_pred_std[sort],
    lla_mean.flatten()[sort] + 2 * lla_pred_std[sort],
    alpha=0.2,
    label="GP uncertainty",
    color="teal",
)


axis[1].fill_between(
    full_dataset.inputs.flatten()[sort],
    np.zeros(full_dataset.inputs.shape[0]),
    lla_pred_std[sort],
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
save_str = "ELLA_dataset={}_M={}".format(
    args.dataset_name, args.num_inducing, args.seed
)

plt.savefig("plots/" + save_str + ".pdf")


ella_std = np.sqrt(ella_var)
lla_std = np.sqrt(lla_var)

KL1 = -np.log(lla_std) + np.log(ella_std) - 0.5 + ((lla_std**2) / (2 * ella_std**2))
KL1 = np.sum(KL1)

KL2 = -np.log(ella_std) + np.log(lla_std) - 0.5 + ((ella_std**2) / (2 * lla_std**2))
KL2 = np.sum(KL2)

MAE = np.mean(np.abs(ella_std - lla_std))


test_metrics = score(
    ella,
    test_loader,
    MetricsRegression,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
    ll_var=ll_var,
)
test_metrics["prior_std"] = args.prior_std
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["KL"] = 0.5 * KL1 + 0.5 * KL2
test_metrics["MAE"] = MAE

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
