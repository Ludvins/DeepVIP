from src.dataset import DVIP_Dataset
from utils import (
    plot_train_test,
    check_data,
    build_plot_name,
    get_parser,
    plot_prior_over_layers,
)
from load_data import SPGP, synthetic, test

import torch
from torch.utils.data import DataLoader

from src.train import predict_prior_samples, train, predict

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from itertools import product

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get experiments variables
epochs = [2000]
bnn_structure = [[10, 10]]
seeds = [0]
regression_coeffs = [20]
lrs = [0.1]
vip_layers = [1]
activations = [torch.tanh]
verbose = 1
freeze_priors = [True, False]
freeze_posteriors = [True, False]
freeze_ll_variances = [True, False]
datasets = ["SPGP", "test", "synthetic"]

for config in product(epochs, bnn_structure, seeds, regression_coeffs, lrs,
                      vip_layers, activations, datasets, freeze_priors,
                      freeze_posteriors, freeze_ll_variances):
    epoch = config[0]
    structure = config[1]
    seed = config[2]
    coeffs = config[3]
    lr = config[4]
    layers = config[5]
    activation = config[6]
    dataset = config[7]
    freeze_prior = config[8]
    freeze_posterior = config[9]
    freeze_ll_variance = config[10]
    # Load data
    if dataset == "SPGP":
        X_train, y_train, X_test, y_test = SPGP()
    elif dataset == "synthetic":
        X_train, y_train, X_test, y_test = synthetic()
    elif dataset == "test":
        X_train, y_train, X_test, y_test = test()

    n_samples, input_dim, output_dim, y_mean, y_std = check_data(
        X_train, y_train, verbose)
    batch_size = n_samples

    # Gaussian Likelihood
    ll = Gaussian()

    # Get VIP layers
    layers = init_layers(X_train,
                         y_train,
                         layers,
                         coeffs,
                         structure,
                         activation=activation,
                         seed=seed,
                         device=device)

    train_dataset = DVIP_Dataset(X_train, y_train)
    test_dataset = DVIP_Dataset(X_test, y_test, normalize=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    predict_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Create DVIP object
    dvip = DVIP_Base(
        ll,
        layers,
        len(train_dataset),
        num_samples=5,
        y_mean=train_dataset.targets_mean,
        y_std=train_dataset.targets_std,
    )

    name_flag = ""
    if freeze_ll_variance:
        dvip.freeze_ll_variance()
        name_flag += "_no_ll_variance_"
    if freeze_posterior:
        dvip.freeze_posterior()
        name_flag += "_no_posterior_"
    if freeze_prior:
        dvip.freeze_prior()
        name_flag += "_no_prior_"

    if freeze_ll_variance and freeze_posterior and freeze_prior:
        continue

    # Define optimizer and compile model
    opt = torch.optim.Adam(dvip.parameters(), lr=lr)

    # Perform training
    train(dvip, train_loader, opt, epochs=epoch)

    # Predict Train and Test
    train_mean, train_var = predict(dvip, predict_loader)
    test_mean, test_var = predict(dvip, test_loader)

    train_prior_samples = predict_prior_samples(dvip, predict_loader)
    test_prior_samples = predict_prior_samples(dvip, test_loader)

    # Create plot title and path
    fig_title, path = build_plot_name(
        config[5],
        structure,
        input_dim,
        output_dim,
        epoch,
        n_samples,
        dataset,
        name_flag,
    )

    plot_train_test((train_mean, train_var), (test_mean, test_var),
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    train_prior_samples,
                    test_prior_samples,
                    title=fig_title,
                    path=path,
                    show=False)
