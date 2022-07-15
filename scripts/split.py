from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, TVIP
from src.layers_init import init_layers
from src.layers import TVIPLayer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Get VIP layers
f = BayesianNN(
    num_samples=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    structure=args.bnn_structure,
    activation=args.activation,
    output_dim=train_dataset.output_dim,
    layer_model=args.bnn_layer,
    dropout=args.dropout,
    fix_random_noise=args.fix_prior_noise,
    zero_mean_prior=args.zero_mean_prior,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

layer = TVIPLayer(
    f,
    num_regression_coeffs=args.regression_coeffs,
    input_dim=train_dataset.input_dim,
    output_dim=train_dataset.output_dim,
    add_prior_regularization=args.prior_kl,
    mean_function=None,
    q_mu_initial_value=0,
    log_layer_noise=-5,
    q_sqrt_initial_value=1,
    dtype=args.dtype,
    device=args.device,
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Create DVIP object
dvip = TVIP(
    likelihood=args.likelihood,
    layer=layer,
    num_data=len(train_dataset),
    bb_alpha=args.bb_alpha,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

# Set the number of training samples to generate
dvip.num_samples = 10
# Train the model
start = timer()
loss = fit(
    dvip,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()

import matplotlib.pyplot as plt

a = np.arange(len(loss) // 3, len(loss))
plt.plot(a, loss[len(loss) // 3 :])
plt.show()


dvip.print_variables()

dvip.num_samples = args.num_samples_test

# Test the model
train_metrics = score(
    dvip, train_test_loader, args.metrics, use_tqdm=True, device=args.device
)
test_metrics = score(dvip, test_loader, args.metrics, use_tqdm=True, device=args.device)

print("TRAIN RESULTS: ")
for k, v in train_metrics.items():
    print("\t - {}: {}".format(k, v))

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))


d = {
    **vars(args),
    **{"time": end - start},
    **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df.to_csv(
    path_or_buf="results/" + create_file_name(args) + ".csv",
    encoding="utf-8",
)
