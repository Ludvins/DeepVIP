import numpy as np

import argparse
from utils import experiment, show_unidimensional_results
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
elif args.dataset == "synthetic":
    X_train, y_train, X_test, y_test = synthetic()

epochs = 20000
vip_layers = 1
bnn_structure = [10, 10]

dvip = experiment(X_train,
                  y_train,
                  vip_layers=vip_layers,
                  structure=bnn_structure,
                  epochs=epochs,
                  seed=0,
                  verbose=1)

mean, var = dvip.predict_y(X_train)

dims = np.concatenate(
    ([X_train.shape[1]], np.ones(vip_layers, dtype=int) * y_train.shape[1]))

dims_name = "-".join(map(str, dims))
bnn_name = "-".join(map(str, bnn_structure))
path = "plots/{}_layers={}_bnn={}_epochs={}_batchsize={}.png".format(
    args.dataset, dims_name, bnn_name, epochs, X_train.shape[0])
fig_title = "Layers: {}  BNN: {}".format(dims_name, bnn_name)

show_unidimensional_results(X_train,
                            y_train,
                            mean,
                            var,
                            path=None,
                            fig_title=fig_title,
                            show=True)
