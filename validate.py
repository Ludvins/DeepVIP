from src.dataset import Boston_Dataset, DVIP_Dataset, Energy_Dataset, Training_Dataset, Test_Dataset
from utils import (
    plot_train_test,
    build_plot_name,
    plot_prior_over_layers,
)
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader

from src.train import predict_prior_samples, train, predict, test
from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers

from sklearn.model_selection import KFold
from process_flags import manage_experiment_configuration
from torch.utils.data import random_split

args = manage_experiment_configuration()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
vars(args)["device"] = device

dataset = Boston_Dataset()

results_dict = {"nelbo": 0.0, "rmse": 0.0, "nll": 0.0}

n_splits = 10
kfold = KFold(n_splits, shuffle=True, random_state=42)

for train_indexes, test_indexes in kfold.split(dataset.inputs):
    train_dataset = Training_Dataset(dataset.inputs[train_indexes],
                                     dataset.targets[train_indexes])
    test_dataset = Test_Dataset(dataset.inputs[test_indexes],
                                dataset.targets[test_indexes])

    # Get VIP layers
    layers = init_layers(train_dataset.inputs, train_dataset.output_dim,
                         **vars(args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)

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
    #dvip.freeze_prior()

    # Define optimizer and compile model
    opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

    # Perform training
    train_metrics = train(
        dvip,
        train_loader,
        opt,
        val_generator=val_loader,
        #scheduler,
        epochs=args.epochs)
    test_metrics = test(dvip, val_loader)

    print("TRAINING RESULTS: ")
    print("\t - NELBO: {}".format(train_metrics["nelbo"]))
    print("\t - NLL: {}".format(train_metrics["nll"]))
    print("\t - RMSE: {}".format(train_metrics["rmse"]))

    print("TEST RESULTS: ")
    print("\t - NELBO: {}".format(test_metrics["nelbo"]))
    print("\t - NLL: {}".format(test_metrics["nll"]))
    print("\t - RMSE: {}".format(test_metrics["rmse"]))

    results_dict["nelbo"] += test_metrics["nelbo"]
    results_dict["rmse"] += test_metrics["rmse"]
    results_dict["nll"] += test_metrics["nll"]

print("TEST RESULTS: ")
print("\t - NELBO: {}".format(results_dict["nelbo"] / n_splits))
print("\t - NLL: {}".format(results_dict["nll"] / n_splits))
print("\t - RMSE: {}".format(results_dict["rmse"] / n_splits))
""" 
    # Predict Train and Test
    train_mean, train_var = predict(dvip, predict_loader)
    train_prediction_mean, train_prediction_var = dvip.get_predictive_results(
        train_mean, train_var)

    # Change MC samples for test
    dvip.num_samples = args.num_samples_test
    test_mean, test_var = predict(dvip, test_loader)
    test_prediction_mean, test_prediction_var = dvip.get_predictive_results(
        test_mean, test_var)

    print(args.X_train.shape)
    print(train_mean.shape)
    import matplotlib.pyplot as plt

    # plot it
    fig = plt.figure(figsize=(20, 10))
    axs = []
    axs.append(fig.add_subplot(3, 5, 1))
    axs.append(fig.add_subplot(3, 5, 2))
    axs.append(fig.add_subplot(3, 5, 3))
    axs.append(fig.add_subplot(3, 5, 4))

    axs.append(fig.add_subplot(3, 5, 6))
    axs.append(fig.add_subplot(3, 5, 7))
    axs.append(fig.add_subplot(3, 5, 8))
    axs.append(fig.add_subplot(3, 5, 9))

    axs.append(fig.add_subplot(3, 5, 11))
    axs.append(fig.add_subplot(3, 5, 12))
    axs.append(fig.add_subplot(3, 5, 13))
    axs.append(fig.add_subplot(3, 5, 14))

    axs.append(fig.add_subplot(1, 5, 5))

    for i in range(args.X_train.shape[1]):
        axs[i].scatter(args.X_train[:, i],
                       args.y_train,
                       color="teal",
                       label="Original data",
                       s=1.5)
        axs[i].scatter(args.X_train[:, i],
                       train_mean.flatten(),
                       color="darkorange",
                       label="Prediction",
                       s=1.5)
        axs[i].legend()
    plt.savefig("2_layers1-1.svg", format="svg")
    plt.savefig("2_layers1-1.png", format="png")
    plt.show()
 """