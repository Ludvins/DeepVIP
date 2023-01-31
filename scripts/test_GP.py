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
from utils.pytorch_learning import fit_map, fit
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


Z = kmeans2(train_dataset.inputs, 20, minit='points', seed = args.seed)[0]

print(train_dataset.inputs.dtype)

def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 10, device = args.device, dtype = args.dtype), 
        torch.nn.Tanh(),
        torch.nn.Linear(10, 10, device = args.device, dtype = args.dtype), 
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1, device = args.device, dtype = args.dtype)
    )
f = get_model()
setattr(f, 'output_size', 1)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.lr)
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
    iterations=1000,
    device=args.device,
)
end = timer()

X = test_dataset.inputs
backend = BackPackInterface(f, "regression")
Jx, _ = backend.jacobians(x = torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype))
Jz, _ = backend.jacobians(x = torch.tensor(test_dataset.inputs, device = args.device, dtype = args.dtype))


prior_std = torch.tensor(2.4607, dtype = args.dtype)
ll_std = torch.tensor(0.1965, dtype = args.dtype)
Kxx = torch.einsum("nds, mds -> dnm", Jx, Jx)
Kzz = torch.einsum("nds, mds -> dnm", Jz, Jz)
Kxz = torch.einsum("nds, mds -> dnm", Jx, Jz)

inv = torch.inverse(
    Kxx + ll_std**2/prior_std**2 * torch.eye(Kxx.shape[1], dtype = args.dtype).unsqueeze(0))
KLLA = prior_std**2 * (Kzz - Kxz.transpose(1, 2) @ inv @Kxz)

import matplotlib.pyplot as plt


fig, axis = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2,1]})

axis[0].scatter(train_dataset.inputs, train_dataset.targets, label = "Training points")
#plt.scatter(test_dataset.inputs, test_dataset.targets, label = "Test points")




f_mu = f(torch.tensor(X, dtype = torch.float64, device=args.device))
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_var = torch.diagonal(KLLA, dim1 = 1, dim2 = 2)
pred_std = np.sqrt(f_var + ll_std**2).flatten().detach().cpu().numpy()


sort = np.argsort(X.flatten())

axis[0].plot(X.flatten()[sort], f_mu.flatten()[sort], label = "Predictions")
axis[0].fill_between(X.flatten()[sort],
                 f_mu.flatten()[sort] - 2*pred_std[sort],
                  f_mu.flatten()[sort] + 2*pred_std[sort],
                  alpha = 0.1,
                  label = "GP_LA uncertainty")


axis[0].legend()

axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 pred_std[sort],
                  alpha = 0.1,
                  label = "GP uncertainty (std)")
axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 ll_std.detach().cpu().numpy(),
                  alpha = 0.1,
                  label = "Likelihood uncertainty (std)")


axis[1].legend()
plt.show()