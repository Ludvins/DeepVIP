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


ella = ELLA_Regression(
    create_ad_hoc_mlp(f),
    f.output_size,
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=args.prior_std,
    log_variance = np.log(ll_var),
    seed=args.seed,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,
)



ll_vars = [-0.5, -0.4, -0.3, -0.2]
prior_stds = np.linspace(0.01, 1, 5)




best_score = np.inf
best_ll_var = None
best_prior_std = None

iters = tqdm(range(len(ll_vars) * len(prior_stds)), unit = " configuration")
iters.set_description("Finding optimal hyper-parameters ")

for i in iters:

    ll_var = ll_vars[i // len(prior_stds)]
    prior_std = prior_stds[i%len(prior_stds)]

    ella.prior_std = torch.tensor(prior_std, device=args.device, dtype=args.dtype)

    if i % len(prior_stds) == 0:
        ella.log_variance = torch.tensor(ll_var, device = args.device, dtype = args.dtype)

        ella.fit_loader(
            torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
            torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
            train_loader,
            verbose = False,
        )

    test_metrics = score(
        ella,
        val_loader,
        MetricsRegression,
        use_tqdm=False,
        device=args.device,
        dtype=args.dtype,
    )

    if test_metrics["NLL"] < best_score:
        best_score = test_metrics["NLL"]
        best_ll_var = ll_var
        best_prior_std = prior_std

save_str = "ELLA_dataset={}_M={}".format(
    args.dataset_name, args.num_inducing, args.seed
)

ella.prior_std = torch.tensor(best_prior_std, device=args.device, dtype=args.dtype)

ella.log_variance = torch.tensor(best_ll_var, device = args.device, dtype = args.dtype)

ella.fit_loader(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
    train_loader,
    verbose = True,
)
        
test_metrics = score(
        ella,
        test_loader,
        MetricsRegression,
        use_tqdm=True,
        device=args.device,
        dtype=args.dtype,
    )



test_metrics["prior_std"] = ella.prior_std.detach().numpy()
test_metrics["log_variance"] = ella.log_variance.detach().numpy()
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
