from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
import tqdm 

from scipy.cluster.vq import kmeans2
sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map
from scripts.filename import create_file_name
from src.generative_functions import *
from laplace import Laplace
from utils.models import get_mlp
from utils.dataset import get_dataset

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


f = get_mlp(train_dataset.inputs.shape[1], train_dataset.targets.shape[1], [50, 50],
            torch.nn.Tanh, device =  args.device, dtype = args.dtype)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr)
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
    iterations=args.MAP_iterations,
    device=args.device,
)
end = timer()

# 'all', 'subnetwork' and 'last_layer'
subset = "all"
# 'full', 'kron', 'lowrank' and 'diag'
hessian = "full"
X = test_dataset.inputs
la = Laplace(f, 'regression', subset_of_weights=subset, hessian_structure=hessian)
la.fit(train_loader)

log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in tqdm.tqdm(range(args.iterations)):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
    
print(np.sqrt(1/np.exp(log_prior.detach().numpy())))
print(np.exp(log_sigma.detach().numpy()))

    
import matplotlib.pyplot as plt


fig, axis = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2,1]}, figsize=(16, 10))

axis[0].scatter(train_dataset.inputs, train_dataset.targets, label = "Training points", 
                color = "black", alpha = 0.9)

X = np.concatenate([train_dataset.inputs, test_dataset.inputs], 0)

f_mu, f_var = la(torch.tensor(X, dtype = torch.float64, device=args.device))
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)


sort = np.argsort(X.flatten())


axis[0].plot(X.flatten()[sort], f_mu.flatten()[sort], label = "Predictions")
axis[0].fill_between(X.flatten()[sort],
                 f_mu.flatten()[sort] - 2*pred_std[sort],
                  f_mu.flatten()[sort] + 2*pred_std[sort],
                  alpha = 0.2,
                  label = "Laplace uncertainty")


axis[0].legend()

axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 pred_std[sort],
                  alpha = 0.2,
                  label = "Laplace uncertainty (std)")
axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 la.sigma_noise.item(),
                  alpha = 0.2,
                  label = "Likelihood uncertainty (std)")

axis[0].xaxis.set_tick_params(labelsize=20)
axis[0].yaxis.set_tick_params(labelsize=20)
axis[1].xaxis.set_tick_params(labelsize=20)
axis[1].yaxis.set_tick_params(labelsize=20)
#axis[2].xaxis.set_tick_params(labelsize=20)
#axis[2].yaxis.set_tick_params(labelsize=20)
axis[0].legend(prop={'size': 14}, loc = 'upper left')
axis[1].legend(prop={'size': 14}, loc = 'upper left')

axis[0].set_ylim(-3, 3)

plt.savefig("Laplace_subset={}_hessian={}.pdf".format(subset, hessian), format="pdf")


plt.show()