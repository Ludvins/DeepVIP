import numpy as np

from src.generative_models import get_bn
from itertools import product
import argparse
from setup import experiment
from load_data import *
import pandas as pd

# Parse dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default="SPGP",
                    help='Dataset to use (SPGP, synthethic or boston)')

args = parser.parse_args()

# Load data
if args.dataset == "SPGP":
    X_train, y_train, X_test, y_test = SPGP()
elif args.dataset == "boston":
    X_train, y_train, X_test, y_test = boston()
elif args.dataset == "synthethic":
    X_train, y_train, X_test, y_test = synthetic()

df = pd.DataFrame(columns=[
    "VIP Layers", "BNN Layers", "Epochs", "nelbo", "rmse", "nll", "seed"
])

epochs = 20000
vip_layers = np.arange(1, 5)
bn_structures = [[10]]
seeds = np.arange(0, 1)

combs = list(product(vip_layers, bn_structures, seeds))

for i, (vip_layers, bnn_structure, seed) in enumerate(combs):
    print("Experiment {} out of {}.".format(i, len(combs)))
    print("VIP Layers: {}, BNN: {}, seed: {}".format(vip_layers, bnn_structure,
                                                     seed))
    dvip = experiment(X_train,
                      y_train,
                      vip_layers=vip_layers,
                      structure=bnn_structure,
                      epochs=epochs,
                      seed=seed,
                      verbose=0)

    metrics = dvip.predict_y(X_test)

    metrics = {
        "VIP Layers": vip_layers,
        "BNN Layers": bnn_structure,
        "Epochs": epochs,
        "nelbo": metrics["nelbo"],
        "rmse": metrics["rmse"],
        "nll": metrics["nll"],
        "seed": seed
    }
    df = df.append(metrics, ignore_index=True)

df.to_csv("synthetic_results.csv")