from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

from scipy.cluster.vq import kmeans2
sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map
from scripts.filename import create_file_name
from src.generative_functions import *
from laplace import Laplace

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


Z = kmeans2(train_dataset.inputs, args.num_inducing, minit='points', seed = args.seed)[0]

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


X = test_dataset.inputs
la = Laplace(f, 'regression', subset_of_weights='all', hessian_structure='full')
la.fit(train_loader)

#la.optimize_prior_precision(method='CV', val_loader=train_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(100):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
    
    
print("Optimal prior std: ", torch.sqrt(torch.exp(log_prior)**-1))
print("Optimal noise std: ", torch.exp(log_sigma))

import matplotlib.pyplot as plt


fig, axis = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2,1]})

axis[0].scatter(train_dataset.inputs, train_dataset.targets, label = "Training points")


X = test_dataset.inputs

f_mu, f_var = la(torch.tensor(X, dtype = torch.float64, device=args.device))
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)


sort = np.argsort(X.flatten())


axis[0].plot(X.flatten()[sort], f_mu.flatten()[sort], label = "Predictions")
axis[0].fill_between(X.flatten()[sort],
                 f_mu.flatten()[sort] - 2*pred_std[sort],
                  f_mu.flatten()[sort] + 2*pred_std[sort],
                  alpha = 0.1,
                  label = "Laplace uncertainty")


axis[0].legend()

axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 pred_std[sort],
                  alpha = 0.1,
                  label = "Laplace uncertainty (std)")
axis[1].fill_between(X.flatten()[sort],
                 np.zeros(X.shape[0]),
                 la.sigma_noise.detach().cpu().numpy(),
                  alpha = 0.1,
                  label = "Likelihood uncertainty (std)")


axis[1].legend()
plt.show()