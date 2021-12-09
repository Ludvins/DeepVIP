from src.dataset import DVIP_Dataset

from load_data import SPGP, synthetic, test, boston

import torch
from torch.utils.data import DataLoader
import sys
from process_flags import check_data, manage_experiment_configuration
from src.train import train, test
import argparse

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from itertools import product
import pandas as pd


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get experiments variables
py_file = f"experiments/{sys.argv[1]}.py"
print(py_file)
exec(open(py_file).read())

df = None

for configuration in product_dict(**params_dict):
    config = argparse.Namespace(**configuration)
    manage_experiment_configuration(config)

    FLAGS = vars(config)
    FLAGS["device"] = device

    check_data(config)

    # Gaussian Likelihood
    ll = Gaussian()

    # Get VIP layers
    layers = init_layers(**vars(config))

    train_dataset = DVIP_Dataset(config.X_train, config.y_train)
    test_dataset = DVIP_Dataset(config.X_test, config.y_test, normalize=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    predict_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Create DVIP object
    dvip = DVIP_Base(
        ll,
        layers,
        len(train_dataset),
        num_samples=1,
        y_mean=train_dataset.targets_mean,
        y_std=train_dataset.targets_std,
    )

    # Define optimizer and compile model
    opt = torch.optim.Adam(dvip.parameters(), lr=config.lr)

    # Perform training
    train_metrics = train(dvip, train_loader, opt, epochs=config.epochs)

    dvip.num_samples = 200

    test_metrics = test(dvip, test_loader, device=device)
    metrics = {
        "train_nelbo": train_metrics["nelbo"].detach().numpy(),
        "train_rmse": train_metrics["rmse"].detach().numpy(),
        "train_nll": train_metrics["nll"].detach().numpy(),
        "test_nelbo": test_metrics["nelbo"].detach().numpy(),
        "test_rmse": test_metrics["rmse"].detach().numpy(),
        "test_nll": test_metrics["nll"].detach().numpy(),
    }
    if df is None:
        df = pd.DataFrame.from_dict({
            **configuration,
            **metrics
        },
                                    orient="index")
        df = df.transpose()

    else:
        df = df.append({**configuration, **metrics}, ignore_index=True)

print(df)
df.to_csv("results/{}.csv".format(sys.argv[1]), index=False)