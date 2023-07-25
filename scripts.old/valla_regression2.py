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
from src.valla import VaLLARegressionRBF, VaLLARegression
from src.lla import GPLLA
from src.likelihood import Gaussian
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from utils.metrics import MetricsRegression

import matplotlib.pyplot as plt
args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

args.dataset = get_dataset(args.dataset_name)
train_dataset, full_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle = True)
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


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()


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


Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

valla = VaLLARegression(
    create_ad_hoc_mlp(f),
    Z,
    prior_std=args.prior_std,
    num_data=train_dataset.inputs.shape[0],
    output_dim=1,
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    alpha = args.bb_alpha,
    device=args.device,
    dtype=args.dtype,
)

# if args.freeze_ll:
#     sparseLA.freeze_ll()


valla.print_variables()

opt = torch.optim.Adam(valla.parameters(recurse=False), lr=args.lr)

start = timer()
loss = fit(
    valla,
    full_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

plt.plot(loss)
plt.show()

valla.print_variables()


full_loader = DataLoader(full_dataset, batch_size=args.batch_size)

plt.rcParams["pdf.fonttype"] = 42
fig, axis = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]}, figsize=(16, 10))


axis[0].scatter(train_dataset.inputs, train_dataset.targets, label="MAP Training Points")
axis[0].scatter(test_dataset.inputs, test_dataset.targets, label="Additional Training Points")


Z = valla.inducing_locations.detach().cpu().numpy()

valla.forward(valla.inducing_locations)

valla_mean, valla_var = forward(valla, full_loader)
valla_std = np.sqrt(valla_var).flatten()

valla_pred_var = valla_var + valla.log_variance.exp().detach().cpu().numpy()
valla_pred_std = np.sqrt(valla_pred_var).flatten()


sort = np.argsort(full_dataset.inputs.flatten())


axis[0].plot(
    full_dataset.inputs.flatten()[sort],
    valla_mean.flatten()[sort],
    label="Predictions",
    color="black",
)
axis[0].fill_between(
    full_dataset.inputs.flatten()[sort],
    valla_mean.flatten()[sort] - 2 * valla_pred_std[sort],
    valla_mean.flatten()[sort] + 2 * valla_pred_std[sort],
    alpha=0.2,
    label="VaLLA uncertainty",
    color="orange",
)

m = f(valla.inducing_locations).flatten().detach().cpu().numpy()

xlims = axis[0].get_xlim()

axis[0].scatter(
    valla.inducing_locations.detach().cpu().numpy(),
    m,
    label="Inducing locations",
    color="tomato",
)

axis[1].fill_between(
    full_dataset.inputs.flatten()[sort],
    np.zeros(full_dataset.inputs.shape[0]),
    valla_pred_std[sort],
    alpha=0.2,
    label="VaLLA uncertainty (std)",
    color="orange",
)
axis[1].fill_between(
    full_dataset.inputs.flatten()[sort],
    np.zeros(full_dataset.inputs.shape[0]),
    valla_pred_std[sort] * 0 + np.sqrt(valla.log_variance.exp().detach().cpu().numpy()),
    alpha=0.2,
    label="Likelihood uncertainty (std)",
    color="teal",
)


axis[0].set_xlim(left=xlims[0], right=xlims[1])

inducing_history = np.stack(valla.inducing_history)
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, inducing_history.shape[1]))

axis[0].legend()
axis[1].legend()

axis[0].set_title("Predictive distribution")
axis[1].set_title("Uncertainty decomposition")


axis[0].xaxis.set_tick_params(labelsize=20)
axis[0].yaxis.set_tick_params(labelsize=20)
axis[1].xaxis.set_tick_params(labelsize=20)
axis[1].yaxis.set_tick_params(labelsize=20)

axis[0].legend(prop={"size": 14}, loc="upper left")
axis[1].legend(prop={"size": 14}, loc="upper left")

plt.show()
save_str = "VaLLA_dataset={}_M={}".format(
    args.dataset_name, args.num_inducing, args.seed
)


plt.savefig("plots/" + save_str + ".pdf")


test_metrics = score(
    valla,
    test_loader,
    MetricsRegression,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
    ll_var=valla.log_variance.exp().detach(),
)
test_metrics["prior_std"] = args.prior_std
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["Classes"] = args.sub_classes

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
