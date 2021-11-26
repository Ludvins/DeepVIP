import numpy as np

from utils import (
    plot_train_test,
    check_data,
    build_plot_name,
    plot_prior_over_layers,
)
from load_data import SPGP, synthetic
import tensorflow as tf

from input_parser import parser

from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.generative_models import GaussianSampler
from src.train import train, predict

# Parse dataset
parser = parser()
args = parser.parse_args()

# Load data
if args.dataset == "SPGP":
    X_train, y_train, X_test, y_test = SPGP()
elif args.dataset == "synthetic":
    X_train, y_train, X_test, y_test = synthetic()

# Get experiments variables
epochs = args.epochs
bnn_structure = args.bnn_structure
seed = args.seed
regression_coeffs = args.regression_coeffs
lr = args.lr
verbose = args.verbose
warmup = args.warmup

if len(args.vip_layers) == 1:
    vip_layers = args.vip_layers[0]

if args.activation == "tanh":
    activation = tf.keras.activations.tanh
elif args.activation == "relu":
    activation = tf.keras.activations.relu

# Set eager execution
tf.config.run_functions_eagerly(args.eager)

n_samples, input_dim, output_dim, y_mean, y_std = check_data(
    X_train, y_train, verbose)
batch_size = args.batch_size or n_samples

# Gaussian Likelihood
ll = Gaussian()

# Define the noise sampler
noise_sampler = GaussianSampler(seed)

# Get VIP layers
layers = init_layers(
    X_train,
    y_train,
    vip_layers,
    regression_coeffs,
    bnn_structure,
    activation=activation,
    noise_sampler=noise_sampler,
    trainable_parameters=True,
    trainable_prior=True,
    seed=seed,
)

# Create DVIP object
dvip = DVIP_Base(
    ll,
    layers,
    num_data=n_samples,
    num_samples=1,
    y_mean=y_mean,
    y_std=y_std,
    warmup_iterations=warmup,
)

# Define optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1,
                                                             decay_steps=100,
                                                             decay_rate=0.96,
                                                             staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, (y_train - y_mean) / y_std))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

tf.config.run_functions_eagerly(False)

train(
    dvip,
    train_dataset,
    optimizer=opt,
    epochs=args.epochs,
    batch_size=args.batch_size,
)

#dvip.num_samples = 100

# Predict Train and Test
mean_pred, var_pred = predict(dvip, train_dataset, batch_size=args.batch_size)
test_mean_pred, test_var_pred = predict(dvip,
                                        test_dataset,
                                        batch_size=args.batch_size)

# Create plot title and path
fig_title, path = build_plot_name(
    vip_layers,
    bnn_structure,
    input_dim,
    output_dim,
    epochs,
    n_samples,
    args.dataset,
    args.name_flag,
)

plot_train_test(
    (mean_pred, var_pred),
    (test_mean_pred, test_var_pred),
    X_train,
    y_train,
    X_test,
    y_test,
    title=fig_title,
    path=path,
)
