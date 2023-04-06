from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

from scipy.cluster.vq import kmeans2
sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit, predict
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import SparseLA
from src.likelihood import Bernoulli
from src.backpack_interface import BackPackInterface
from utils.models import get_mlp
from utils.dataset import get_dataset, Test_Dataset
from torchsummary import summary
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




f = get_mlp(train_dataset.inputs.shape[1], train_dataset.targets.shape[1], 
            [50, 50], torch.nn.Tanh, torch.nn.Sigmoid,
            args.device, args.dtype)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
criterion = torch.nn.BCELoss()

# Set the number of training samples to generate
# Train the model
start = timer()

loss = fit_map(
    f,
    train_loader,
    opt,
    criterion = criterion,
    use_tqdm=True,
    return_loss=True,
    iterations=args.MAP_iterations,
    device=args.device,
)
end = timer()

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

sparseLA = SparseLA(
    f[:-1].forward,
    Z, 
    alpha = args.bb_alpha,
    prior_std=1,
    likelihood=Bernoulli(device=args.device, 
                        dtype = args.dtype), 
    num_data = train_dataset.inputs.shape[0],
    output_dim = 1,
    backend = BackPackInterface(f, "classification"),
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,)

if args.freeze_ll:
    sparseLA.freeze_ll()
    
    
sparseLA.print_variables()

opt = torch.optim.Adam(sparseLA.parameters(), lr=args.lr)

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

sparseLA.print_variables()



import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['pdf.fonttype'] = 42
fig, axis = plt.subplots(3, 2, figsize=(15, 15))


axis[0][0].scatter(train_dataset.inputs, train_dataset.targets, alpha = 0.8, label = "Training Dataset")
axis[0][0].set_title("Training Dataset")
xlims = axis[0][0].get_xlim()
ylims = axis[0][0].get_ylim()



mean, var = predict(sparseLA, test_loader)


plt.show()
