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
from laplace import Laplace
import tqdm

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

numpy.set_printoptions(threshold=sys.maxsize)


# 'all', 'subnetwork' and 'last_layer'
subset = "all"
# 'full', 'kron', 'lowrank' and 'diag'
hessian = "kron"
X = test_dataset.inputs
la = Laplace(f, "regression", subset_of_weights=subset, hessian_structure=hessian)
la.fit(train_loader)

log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
    1, requires_grad=True
)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-3)
for i in tqdm.tqdm(range(args.iterations)):
    hyper_optimizer.zero_grad()
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
log_variance = 2*log_sigma.detach().numpy().item()



y_mean = torch.tensor(train_dataset.targets_mean, device=args.device)
y_std = torch.tensor(train_dataset.targets_std, device=args.device)



def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if args.dtype != X.dtype:
            X = X.to(args.dtype)
        if args.dtype != y.dtype:
            y = y.to(args.dtype)

        Fmean, Fvar = la._glm_predictive_distribution(X)  # Forward pass
        Fvar = Fvar + np.exp(log_variance)
        return 0, Fmean * y_std + y_mean, Fvar * y_std**2


la.test_step = test_step

save_str = "LLA_dataset={}_{}_{}".format(
    args.dataset_name, subset, hessian
)


test_metrics = score(
    la,
    test_loader,
    MetricsRegression,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)
test_metrics["prior_std"] = prior_std
test_metrics["log_variance"] = log_variance
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["subset"] = subset
test_metrics["hessian"] = hessian

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
