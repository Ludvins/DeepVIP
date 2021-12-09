from src.dataset import DVIP_Dataset
from utils import (
    plot_train_test,
    build_plot_name,
    plot_prior_over_layers,
)
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.train import predict_prior_samples, train, predict, test
from src.train import predict_prior_samples, train, predict, test

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers

from process_flags import manage_experiment_configuration

args = manage_experiment_configuration()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

# Get VIP layers
layers = init_layers(**vars(args))

train_dataset = DVIP_Dataset(args.X_train, args.y_train)
test_dataset = DVIP_Dataset(args.X_test, args.y_test, normalize=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Instantiate Likelihood
ll = Gaussian()

# Create DVIP object
dvip = DVIP_Base(
    ll,
    layers,
    len(train_dataset),
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
)

if args.freeze_prior:
    dvip.freeze_prior()
if args.freeze_posterior:
    dvip.freeze_posterior()

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.996)
scheduler = None

# Perform training
train_metrics = train(dvip,
                      train_loader,
                      opt,
                      scheduler,
                      epochs=args.epochs,
                      device=args.device)

# Change MC samples for test
dvip.num_samples = args.num_samples_test

test_metrics = test(dvip, test_loader, device=args.device)

dvip.print_variables()

print("TRAINING RESULTS: ")
print("\t - NELBO: {}".format(train_metrics["nelbo"]))
print("\t - NLL: {}".format(train_metrics["nll"]))
print("\t - RMSE: {}".format(train_metrics["rmse"]))

print("TEST RESULTS: ")
print("\t - NELBO: {}".format(test_metrics["nelbo"]))
print("\t - NLL: {}".format(test_metrics["nll"]))
print("\t - RMSE: {}".format(test_metrics["rmse"]))