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
    iterations=1000,
    device=args.device,
)
end = timer()

prior_std = torch.tensor(2.4607, dtype = args.dtype)
ll_std = torch.tensor(0.1965, dtype = args.dtype)

M = 50
K_eigh = 30

rng = np.random.default_rng(1234)
indexes = rng.choice(np.arange(train_dataset.inputs.shape[0]), M)
Xz = train_dataset.inputs[indexes]
yz = train_dataset.targets[indexes]

backend = BackPackInterface(f, "regression")
Js, _ = backend.jacobians(x = torch.tensor(Xz, device = args.device, dtype = args.dtype))

K = torch.einsum("nds, mds -> dnm", Js, Js)


L, V = torch.linalg.eigh(K)

L = L[:, -K_eigh:]
V = V[:, :, -K_eigh:]

v = torch.einsum("mds, dmk -> dsk", Js, V/torch.sqrt(L).unsqueeze(1)).squeeze(0)




import matplotlib.pyplot as plt
fig, axis = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2,1]})

axis[0].scatter(train_dataset.inputs, train_dataset.targets, label = "Training points")
#plt.scatter(test_dataset.inputs, test_dataset.targets, label = "Test points")



Jtrain, _ = backend.jacobians(x = torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype))
phi_train = torch.einsum("mds, sk -> dmk", Jtrain, v).squeeze(0)

G = torch.eye(K_eigh)/prior_std**2
for p in phi_train:
    G = G + p.unsqueeze(-1) @ p.unsqueeze(0) * ll_std **2

X = np.concatenate([train_dataset.inputs, test_dataset.inputs], 0)
J, _ = backend.jacobians(x = torch.tensor(X, device = args.device, dtype = args.dtype))

phi = torch.einsum("mds, sk -> dmk", J, v).squeeze(0)

K = phi @ torch.inverse(G) @ phi.T


f_mu = f(torch.tensor(X, device = args.device, dtype = args.dtype)).detach().cpu().numpy()
f_var = torch.diagonal(K).squeeze().detach().cpu().numpy()
pred_std = np.sqrt(f_var + (ll_std**2).detach().cpu().numpy())


sort = np.argsort(X.flatten())

axis[0].plot(X.flatten()[sort], f_mu.flatten()[sort], label = "Predictions")
axis[0].fill_between(X.flatten()[sort],
                 f_mu.flatten()[sort] - 2*pred_std[sort],
                  f_mu.flatten()[sort] + 2*pred_std[sort],
                  alpha = 0.1,
                  label = "SparseLA uncertainty")

axis[0].scatter(Xz, yz, label = "Nystrom locations")

axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 pred_std[sort],
                  alpha = 0.1,
                  label = "SparseLA uncertainty (std)")

axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 ll_std.detach().cpu().numpy(),
                  alpha = 0.1,
                  label = "Likelihood uncertainty (std)")
axis[0].legend()
axis[1].legend()

plt.show()