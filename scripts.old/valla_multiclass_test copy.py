from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.vq import kmeans2

sys.path.append(".")
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import (
    fit_map_crossentropy,
    fit,
    predict,
    forward,
    score,
    acc_multiclass,
)
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import VaLLAMultiClassSubset, GPLLA
from src.likelihood import GaussianMultiClass, MultiClass
from src.backpack_interface import BackPackInterface

from torch.profiler import profile, record_function, ProfilerActivity

from utils.models import MLP
from utils.dataset import get_dataset, Test_Dataset
from laplace import Laplace

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)


args.dataset = get_dataset(args.dataset_name)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = MLP(
    train_dataset.inputs.shape[1],
    args.dataset.output_dim,
    1,
    100,
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.CrossEntropyLoss()

try:
    f.load_state_dict(torch.load("weights/multiclass_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map_crossentropy(
        f,
        train_loader,
        opt,
        criterion=criterion,
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
        dtype=args.dtype,
    )
    plt.plot(loss)

    plt.show()
    print("MAP Loss: ", loss[-1])
    end = timer()

    train_acc = acc_multiclass(
        f, train_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    test_acc = acc_multiclass(
        f, test_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    print("Train acc: ", train_acc)
    print("Test acc: ", test_acc)

    torch.save(f.state_dict(), "weights/multiclass_weights_" + args.dataset_name)

for param in f.parameters():
    param.requires_grad = False

# Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)

rng = np.random.default_rng(args.seed)
indexes = rng.choice(
    np.arange(train_dataset.inputs.shape[0]), args.num_inducing, replace=False
)
Z = train_dataset.inputs[indexes]
classes = train_dataset.targets[indexes].flatten()

sparseLA = VaLLAMultiClassSubset(
    f,
    Z,
    n_classes_subsampled=2,
    inducing_classes=classes,
    alpha=args.bb_alpha,
    prior_std=args.prior_std,
    likelihood=GaussianMultiClass(device=args.device, dtype=args.dtype),
    num_data=train_dataset.inputs.shape[0],
    output_dim=args.dataset.output_dim,
    backend=BackPackInterface(f, f.output_size),
    fix_inducing_locations=False,
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,
)


sparseLA.print_variables()

opt = torch.optim.Adam(sparseLA.parameters(), lr=args.lr)

path = "weights/valla_multiclass_weights_{}_{}".format(
    str(args.num_inducing), str(args.iterations)
)

import os

if os.path.isfile(path):
    print("Pre-trained weights found")

    sparseLA.load_state_dict(torch.load(path))
else:
    print("Pre-trained weights not found")
    start = timer()

    loss = fit(
        sparseLA,
        train_loader,
        opt,
        use_tqdm=True,
        return_loss=True,
        iterations=args.iterations,
        device=args.device,
    )

    end = timer()
    fig, axis = plt.subplots(3, 1, figsize=(15, 20))
    axis[0].plot(loss)
    axis[0].set_title("Nelbo")

    axis[1].plot(sparseLA.ell_history)
    axis[1].set_title("ELL")

    axis[2].plot(sparseLA.kl_history)
    axis[2].set_title("KL")

    plt.show()
    # plt.clf()

    # torch.save(sparseLA.state_dict(), path)


sparseLA.print_variables()

_, valla_var = forward(sparseLA, test_loader)
