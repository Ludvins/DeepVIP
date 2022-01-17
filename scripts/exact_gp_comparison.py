import matplotlib.pyplot as plt
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
    Synthetic_Dataset,
)
from utils.plotting_utils import build_plot_name, plot_train_test
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, predict, fit_with_metrics

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

device = "cpu"  # torch.device("cuda:0" if use_cuda else "cpu")
vars(args)["device"] = device

# Generate train/test partition using split number
train_indexes, test_indexes = train_test_split(
    np.arange(len(args.dataset)),
    test_size=0.1,
    random_state=2147483647,
)

train_dataset = Training_Dataset(
    args.dataset.inputs[train_indexes],
    args.dataset.targets[train_indexes],
    verbose=False,
)
test_dataset = Test_Dataset(
    args.dataset.inputs,
    args.dataset.targets,
    train_dataset.inputs_mean,
    train_dataset.inputs_std,
)


########## EXACT GP #############

plt.scatter(
    test_dataset.inputs, test_dataset.targets, label="Test points", s=2, alpha=0.8
)
plt.scatter(
    train_dataset.inputs,
    train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean,
    label="Training points",
    s=20.0,
)


def RBF(x, y=None, kernel_amp=1, kernel_length=1):
    if y is None:
        y = x
    x = x / kernel_length
    dist = (x - y.T) ** 2
    return np.exp(-0.5 * dist / kernel_amp ** 2)


def get_exact_GP_predictions(
    x_train, y_train, x_test, y_mean, y_std, kernel_amp, kernel_length, white_noise
):

    Ku = RBF(x_test, x_train, kernel_amp=kernel_amp, kernel_length=kernel_length)
    Kuu = RBF(x_test, kernel_amp=kernel_amp, kernel_length=kernel_length)
    K = RBF(x_train, kernel_amp=kernel_amp, kernel_length=kernel_length)
    K = K + white_noise * np.eye(y_train.shape[0])

    GP_pred_mean = Ku @ np.linalg.inv(K) @ y_train
    GP_pred_mean = GP_pred_mean * y_std + y_mean

    GP_pred_var = Kuu - Ku @ np.linalg.inv(K) @ Ku.T
    GP_pred_std = np.sqrt(np.diag(GP_pred_var)[:, np.newaxis]) * y_std

    return GP_pred_mean, GP_pred_std


# print(RBF(train_dataset.inputs))

sort = np.argsort(test_dataset.inputs.flatten())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Get VIP layers
layers = init_layers(train_dataset.inputs, train_dataset.output_dim, **vars(args))


# Instantiate Likelihood
ll = Gaussian()

dvip = DVIP_Base(
    ll,
    layers,
    len(train_dataset),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
# dvip.freeze_prior()

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


# Train the model
fit_with_metrics(
    dvip,
    train_loader,
    opt,
    epochs=args.epochs,
    device=args.device,
)

dvip.print_variables()


def get_predictive_results(mean, sqrt):

    prediction_mean = np.mean(mean, axis=0)
    prediction_var = np.mean(sqrt ** 2 + mean ** 2, axis=0) - prediction_mean ** 2
    return prediction_mean, np.sqrt(prediction_var)


# Change MC samples for test
dvip.num_samples = args.num_samples_test
test_mean, test_sqrt = predict(dvip, test_loader, device=args.device)
test_prediction_mean, test_prediction_sqrt = get_predictive_results(
    test_mean, test_sqrt
)

plt.plot(
    test_dataset.inputs.flatten()[sort],
    test_prediction_mean[sort],
    label="VIP prediction",
    color="teal",
)
plt.fill_between(
    test_dataset.inputs.flatten()[sort],
    (test_prediction_mean[sort] - test_prediction_sqrt[sort]).flatten(),
    (test_prediction_mean[sort] + test_prediction_sqrt[sort]).flatten(),
    color="teal",
    alpha=0.2,
)

try:
    kernel_amp = np.exp(
        dvip.vip_layers[0].generative_function.log_kernel_amp.detach().numpy()
    )
    kernel_length = np.exp(
        dvip.vip_layers[0].generative_function.log_kernel_length.detach().numpy()
    )
except:
    kernel_amp = 1.0
    kernel_length = 1.0

white_noise = np.exp(dvip.likelihood.log_variance.detach().numpy())


GP_pred_mean, GP_pred_std = get_exact_GP_predictions(
    train_dataset.inputs,
    train_dataset.targets,
    test_dataset.inputs,
    train_dataset.targets_mean,
    train_dataset.targets_std,
    kernel_amp,
    kernel_length,
    white_noise,
)

plt.plot(
    test_dataset.inputs.flatten()[sort],
    GP_pred_mean[sort],
    label="GP prediction",
    color="black",
)
plt.fill_between(
    test_dataset.inputs.flatten()[sort],
    (GP_pred_mean[sort] - GP_pred_std[sort]).flatten(),
    (GP_pred_mean[sort] + GP_pred_std[sort]).flatten(),
    color="gray",
    alpha=0.2,
)


plt.legend()
plt.show()
