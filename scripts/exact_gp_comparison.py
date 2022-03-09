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
from utils.pytorch_learning import fit, predict, fit_with_metrics, predict_prior_samples
from scripts.filename import create_file_name

args = manage_experiment_configuration()

torch.manual_seed(2147483647)

device = "cpu"  # torch.device("cuda:0" if use_cuda else "cpu")
vars(args)["device"] = device


################## DATASET ###################

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

# inputs = rng.standard_normal(300)
inputs = np.loadtxt("data/SPGP_dist/test_inputs")

test_dataset = Test_Dataset(
    inputs[..., np.newaxis],
    inputs_mean=train_dataset.inputs_mean,
    inputs_std=train_dataset.inputs_std,
)

from matplotlib import pyplot as plt

plt.figure(figsize=(18, 10))
# plt.scatter(
#    test_dataset.inputs, test_dataset.targets, label="Test points", s=2, alpha=0.8
# )
plt.scatter(
    train_dataset.inputs,
    train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean,
    label="Training points",
    s=20.0,
)


################## EXACT GP #################


def RBF(x, y=None, kernel_amp=1, kernel_length=1):
    if y is None:
        y = x
    dist = (x - y.T) ** 2
    return kernel_amp ** 2 * np.exp(-0.5 * dist / kernel_length ** 2)


def get_exact_GP_predictions(
    x_train, y_train, x_test, y_mean, y_std, kernel_amp, kernel_length, white_noise
):

    Ku = RBF(x_test, x_train, kernel_amp=kernel_amp, kernel_length=kernel_length)
    Kuu = RBF(x_test, kernel_amp=kernel_amp, kernel_length=kernel_length)
    K = RBF(x_train, kernel_amp=kernel_amp, kernel_length=kernel_length)

    K = K + white_noise * np.eye(y_train.shape[0])
    Kuu = Kuu + white_noise * np.eye(x_test.shape[0])

    K_inv = np.linalg.inv(K)

    GP_pred_mean = Ku @ K_inv @ y_train
    GP_pred_mean = GP_pred_mean * y_std + y_mean

    GP_pred_var = Kuu - Ku @ K_inv @ Ku.T
    GP_pred_std = np.sqrt(np.diag(GP_pred_var))[:, np.newaxis] * y_std

    return GP_pred_mean, GP_pred_std


################ DVIP TRAINING ##############

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Get VIP layers
layers = init_layers(train_dataset.inputs, train_dataset.output_dim, **vars(args))


# Instantiate Likelihood
ll = args.likelihood(device=args.device, dtype = args.dtype)

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


#################### EXACT DVIP #############################


def get_exact_VIP_predictions(x_train, y_train, x_test, y_mean, y_std, white_noise):

    layer = dvip.vip_layers[0]
    genf = layer.generative_function
    # Let S = num_coeffs, D = output_dim and N = num_samples
    # Shape (S, N, D)
    f_train = genf(torch.tensor(x_train)).detach().numpy()
    f_test = genf(torch.tensor(x_test)).detach().numpy()

    # Compute mean value, shape (N, D)
    m_train = np.mean(f_train, axis=0)

    # Compute regresion function, shape (S , N, D)
    phi_train = (f_train - m_train) / np.sqrt(layer.num_coeffs - 1)
    # Shape (N, N, D)
    I = np.eye(x_train.shape[0])
    K_train = np.einsum("snd, smd -> dnm", phi_train, phi_train) + white_noise * I
    K_train_inv = np.transpose(np.linalg.inv(K_train), (1, 2, 0))

    A = np.einsum("snd, nmd -> smd", phi_train, K_train_inv)
    q_mu = np.einsum("snd, nd -> sd", A, y_train - m_train)

    I = np.eye(layer.num_coeffs)
    I_tiled = np.tile(I[..., np.newaxis], [1, 1, layer.output_dim])
    q_sigma = I_tiled - np.einsum("snd, and -> sad", A, phi_train)
    # Shape (S, N, D)

    # Compute mean value, shape (N, D)
    m_test = np.mean(f_test, axis=0)
    # Compute regresion function, shape (S , N, D)
    phi_test = (f_test - m_test) / np.sqrt(layer.num_coeffs - 1)

    mean = m_test + np.einsum("snd,sd->nd", phi_test, q_mu)

    # Compute variance in two steps
    # Compute phi^T Delta = phi^T s_qrt q_sqrt^T
    K = np.einsum("snd,skd->knd", phi_test, q_sigma)
    # Multiply by phi again, using the same points twice
    K = np.einsum("snd,snd->nd", K, phi_test)
    # Add layer noise to variance
    if layer.log_layer_noise is not None:
        K = K + np.exp(layer.log_layer_noise.detach().numpy())

    mean = mean * y_std + y_mean
    sqrt = np.sqrt(K + white_noise) * y_std

    return mean, sqrt


###############################################################


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
test_prior_samples = predict_prior_samples(dvip, test_loader)[0]

sort = np.argsort(test_dataset.inputs.flatten())


plt.plot(
    test_dataset.inputs.flatten()[sort],
    test_prediction_mean[sort],
    label="VIP prediction",
    color="purple",
)
plt.fill_between(
    test_dataset.inputs.flatten()[sort],
    (test_prediction_mean[sort] - 3 * test_prediction_sqrt[sort]).flatten(),
    (test_prediction_mean[sort] + 3 * test_prediction_sqrt[sort]).flatten(),
    color="purple",
    alpha=0.2,
)

for prior_sample in test_prior_samples[:10]:
    plt.plot(
        test_dataset.inputs.flatten()[sort],
        prior_sample.flatten()[sort],
        color="red",
        alpha=0.1,
    )

try:
    kernel_amp = np.exp(
        dvip.vip_layers[0].generative_function.log_kernel_amp.detach().numpy()
    )
    kernel_length = np.exp(
        dvip.vip_layers[0].generative_function.log_kernel_length.detach().numpy()
    )
except:
    kernel_amp = np.exp(0.19156675)
    kernel_length = np.exp(-0.99385499)

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
    (GP_pred_mean[sort] - 3 * GP_pred_std[sort]).flatten(),
    (GP_pred_mean[sort] + 3 * GP_pred_std[sort]).flatten(),
    color="gray",
    alpha=0.2,
)

VIP_pred_mean, VIP_pred_std = get_exact_VIP_predictions(
    train_dataset.inputs,
    train_dataset.targets,
    test_dataset.inputs,
    train_dataset.targets_mean,
    train_dataset.targets_std,
    white_noise,
)

plt.plot(
    test_dataset.inputs.flatten()[sort],
    VIP_pred_mean[sort],
    label="VIP exact prediction",
    color="orange",
)
plt.fill_between(
    test_dataset.inputs.flatten()[sort],
    (VIP_pred_mean[sort] - 3 * VIP_pred_std[sort]).flatten(),
    (VIP_pred_mean[sort] + 3 * VIP_pred_std[sort]).flatten(),
    color="orange",
    alpha=0.2,
)

plt.legend()
plt.savefig("plots/gp_comp_" + create_file_name(args) + ".pdf", dpi=1000)
plt.show()
