from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

from laplace.curvature import AsdlInterface, BackPackInterface
from laplace import Laplace

from scipy.cluster.vq import kmeans2
sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, fit, score
from scripts.filename import create_file_name
from src.generative_functions import *
from src.sparseLA import SparseLA
from src.likelihood import Gaussian

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)




def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(train_dataset.inputs.shape[1], 10, device = args.device, dtype = args.dtype), 
        torch.nn.Tanh(),
        torch.nn.Linear(10, 10, device = args.device, dtype = args.dtype), 
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1, device = args.device, dtype = args.dtype)
    )
f = get_model()
setattr(f, 'output_size', 1)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Set the number of training samples to generate
# Train the model
start = timer()

loss = fit_map(
    f,
    train_loader,
    opt,
    criterion = torch.nn.MSELoss(),
    use_tqdm=True,
    return_loss=True,
    iterations=10000,
    device=args.device,
)
end = timer()

Z = kmeans2(train_dataset.inputs, 20, minit="points", seed=args.seed)[0]
sparseLA = SparseLA(
    f.forward,
    Z, 
    prior_variance_init=1,
    likelihood=Gaussian(device=args.device, dtype = args.dtype), 
    num_data = train_dataset.inputs.shape[0],
    output_dim = 1,
    backend = BackPackInterface(f, "regression"),
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,)

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


print("Last NELBO value: ", loss[-1])

import matplotlib.pyplot as plt

a = np.arange(len(loss) // 3, len(loss))
plt.plot(a, loss[len(loss) // 3 :])
plt.show()


train_metrics = score(
    sparseLA, train_test_loader, args.metrics, use_tqdm=True, device=args.device
)
test_metrics = score(sparseLA, test_loader, args.metrics, use_tqdm=True, device=args.device)

print("TRAIN RESULTS: ")
for k, v in train_metrics.items():
    print("\t - {}: {}".format(k, v))

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))

