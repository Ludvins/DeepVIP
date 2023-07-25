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
from src.ella import ELLA_Regression
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from utils.metrics import MetricsRegression
from tqdm import tqdm
args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

args.dataset = get_dataset(args.dataset_name)
train_dataset, val_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = get_mlp(
    train_dataset.inputs.shape[1],
    train_dataset.targets.shape[1],
    [200, 200, 200],
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)

if args.weight_decay != 0:
    args.prior_std = np.sqrt(1 / (len(train_dataset) * args.weight_decay))


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

save_str = "MAP_dataset={}".format(
    args.dataset_name)


y_mean = torch.tensor(train_dataset.targets_mean, device=args.device)
y_std = torch.tensor(train_dataset.targets_std, device=args.device)

ll_vars = np.linspace(-5, 5, 50)

def get_test_step(ll_variance):

    def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if args.dtype != X.dtype:
            X = X.to(args.dtype)
        if args.dtype != y.dtype:
            y = y.to(args.dtype)

        Fmean = f(X)  # Forward pass
        Fvar = torch.ones_like(Fmean) * ll_variance

        return 0, Fmean * y_std + y_mean, Fvar * y_std**2
    return test_step

best_score = np.inf
best_ll_var = None

iters = tqdm(range(len(ll_vars)), unit = " configuration")
iters.set_description("Finding optimal noise variance ")
for i in iters:
    f.test_step = get_test_step(ll_vars[i])

    test_metrics = score(
        f,
        val_loader,
        MetricsRegression,
        use_tqdm=False,
        device=args.device,
        dtype=args.dtype,
    )

    if test_metrics["NLL"] < best_score:
        best_score = test_metrics["NLL"]
        best_ll_var = ll_vars[i]

test_metrics = score(
    f,
    test_loader,
    MetricsRegression,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)

test_metrics["log_variance"] = np.log(best_ll_var)
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations


df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
