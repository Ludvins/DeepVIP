import numpy as np

from utils import plot_train_test, check_data, build_plot_name
from load_data import SPGP, synthetic, boston
import tensorflow as tf

from parser import get_parser
import pandas as pd
from src.likelihood import Gaussian
from src.dvip import DVIP_Base
from src.layers_init import init_layers
from src.generative_models import GaussianSampler

from tqdm.keras import TqdmCallback
from itertools import product
# Parse dataset
parser = get_parser()
args = parser.parse_args()

# Load data
if args.dataset == "SPGP":
    X_train, y_train, X_test, y_test = SPGP()
elif args.dataset == "boston":
    X_train, y_train, X_test, y_test = boston()
elif args.dataset == "synthethic":
    X_train, y_train, X_test, y_test = synthetic()

df = pd.DataFrame(
    columns=["VIP Layers", "BNN Layers", "Epochs", "nelbo", "rmse", "nll", "seed"]
)

epochs = 20000
vip_layers = np.arange(1, 5)
bn_structures = [[10]]
seeds = np.arange(0, 1)

combs = list(product(vip_layers, bn_structures, seeds))


n_samples, input_dim, output_dim, y_mean, y_std = check_data(X_train, y_train)
batch_size = n_samples

# Gaussian Likelihood
ll = Gaussian()


for i, (vip_layers, bnn_structure, seed) in enumerate(combs):
    print("Experiment {} out of {}.".format(i, len(combs)))
    print("VIP Layers: {}, BNN: {}, seed: {}".format(vip_layers, bnn_structure, seed))
    # Define the noise sampler
    noise_sampler = GaussianSampler(seed)

    # Get VIP layers
    layers = init_layers(
        X_train,
        y_train,
        vip_layers,
        args.regression_coeffs,
        bnn_structure,
        activation=args.activation,
        noise_sampler=noise_sampler,
        seed=seed,
    )

    # Create DVIP object
    dvip = DVIP_Base(ll, layers, input_dim, y_mean=y_mean, y_std=y_std)

    # Define optimizer and compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    dvip.compile(optimizer=opt)

    # Perform training
    dvip.fit(
        X_train,
        (y_train - y_mean) / y_std,  # Provide normalized outputs
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[TqdmCallback(verbose=0)],
    )

    metrics = dvip.evaluate(X_test)

    metrics = {
        "VIP Layers": vip_layers,
        "BNN Layers": bnn_structure,
        "Epochs": epochs,
        "nelbo": metrics["nelbo"],
        "rmse": metrics["rmse"],
        "nll": metrics["nll"],
        "seed": seed,
    }
    df = df.append(metrics, ignore_index=True)

df.to_csv("{}_results.csv".format(args.dataset))
