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
    iterations=10000,
    device=args.device,
)
end = timer()

prior = torch.tensor(np.exp(-5.03566519), dtype = args.dtype)
ll = torch.tensor(np.exp(-1.02793543), dtype = args.dtype)

M = 20
K = 10

rng = np.random.default_rng(1234)
indexes = rng.choice(np.arange(train_dataset.inputs.shape[0]), M)
Xz = train_dataset.inputs[indexes]
yz = train_dataset.targets[indexes]

backend = BackPackInterface(f, "regression")
Js, _ = backend.jacobians(x = torch.tensor(Xz, device = args.device, dtype = args.dtype))

K = torch.einsum("nds, mds -> dnm", Js, Js)


L, V = torch.linalg.eigh(K)
L = L[:, -10:]
V = V[:, :, -10:]


v = torch.einsum("mds, dmk -> dsk", Js, V/torch.sqrt(L).unsqueeze(1)).squeeze(0)




X = test_dataset.inputs
Z = train_dataset.inputs[indexes]
Jx, _ = backend.jacobians(x = torch.tensor(X, device = args.device, dtype = args.dtype))
phi = torch.einsum("mds, sk -> dmk", Jx, v).squeeze(0)

G = torch.eye(10)/prior
for p in phi:
    G = G + p.unsqueeze(-1) @ p.unsqueeze(0) * ll

K = phi @ torch.inverse(G) @ phi.T


f_mu = f(torch.tensor(X, device = args.device, dtype = args.dtype))
f_var = torch.diagonal(K).unsqueeze(-1)


likelihood = Gaussian(ll, dtype = args.dtype, device = args.device)

logp = likelihood.logdensity(f_mu, f_var, torch.tensor(test_dataset.targets, device = args.device, dtype = args.dtype))
logp = torch.mean(logp)

print("TEST RESULTS: ")
print("\t\t - NLL:", -logp.detach().cpu().numpy())
