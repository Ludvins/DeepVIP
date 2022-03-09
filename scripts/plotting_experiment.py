import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from sklearn.model_selection import train_test_split

sys.path.append(".")

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.likelihood import Gaussian
from utils.dataset import (
    Test_Dataset,
    Training_Dataset,
)
from utils.plotting_utils import build_plot_name, plot_train_test
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, fit_with_metrics, predict, predict_prior_samples, score
args = manage_experiment_configuration()

torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(0.1, args.seed + args.split)

# Get VIP layers
layers = init_layers(train_dataset.inputs, args.dataset.output_dim, **vars(args))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Create DVIP object
dvip = DVIP_Base(
    args.likelihood,
    layers,
    len(train_dataset),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


# Train the model
fit_with_metrics(
    dvip,
    train_loader,
    opt,
    args.metrics,
    val_generator=test_loader,
    epochs=args.epochs,
    device=args.device,
)

dvip.print_variables()

# Predict Train and Test
dvip.num_samples = args.num_samples_train
train_mean, train_var = predict(dvip, train_test_loader, device=args.device)


def get_predictive_results(mean, var):

    prediction_mean = np.mean(mean, axis=0)
    prediction_var = np.mean(var + mean ** 2, axis=0) - prediction_mean ** 2
    return prediction_mean, prediction_var


train_prediction_mean, train_prediction_var = get_predictive_results(
    train_mean, train_var
)

# Change MC samples for test
dvip.num_samples = args.num_samples_test
test_mean, test_var = predict(dvip, test_loader, device=args.device)
test_prediction_mean, test_prediction_var = get_predictive_results(test_mean, test_var)

dvip.num_samples = args.num_samples_train
dvip.train()
train_prior_samples = predict_prior_samples(dvip, train_test_loader)
dvip.eval()
test_prior_samples = predict_prior_samples(dvip, test_loader)

# Create plot title and path
fig_title, path = build_plot_name(**vars(args))

plot_train_test(
    train_mixture_means=train_mean,
    train_prediction_mean=train_prediction_mean,
    train_prediction_sqrt=np.sqrt(train_prediction_var),
    test_mixture_means=test_mean,
    test_prediction_mean=test_prediction_mean,
    test_prediction_sqrt=np.sqrt(test_prediction_var),
    X_train=train_test_dataset.inputs.flatten(),
    y_train=train_test_dataset.targets.flatten(),
    X_test=test_dataset.inputs.flatten(),
    y_test=test_dataset.targets.flatten(),
    train_prior_samples=train_prior_samples[-1][:10],
    test_prior_samples=test_prior_samples[-1][:10],
    title=fig_title,
    path=path,
    show=args.show,
)
