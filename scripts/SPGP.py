from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2
import gpytorch

sys.path.append(".")

from src.fvi import FVI, FVI2, FVI3, FVI4, FVI5
from src.layers_init import init_layers, init_layers_tvip
from src.layers import TVIPLayer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

#args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = torch.device("cpu")
torch.manual_seed(args.seed)
args.dtype = torch.float32

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

Z = kmeans2(train_dataset.inputs, args.num_inducing, minit='points', seed = args.seed)[0]

f = GP(
    input_dim=1,
    output_dim=1,
    inner_layer_dim=200,
    kernel_amp=1,
    kernel_length=1,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)

X = torch.tensor(test_dataset.inputs, dtype =args.dtype)

class Kernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **args):
        return f.GP_cov(x1, x2)

class Mean(gpytorch.means.Mean):                                                                                                                                                                        

    def forward(self, x):
        return f.GP_mean(x).T

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0), dtype = args.dtype, mean_init_std=1e-10)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.f = f
        self.mean_module = Mean()
        self.covar_module = Kernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



model = GPModel(inducing_points=torch.tensor(Z, dtype = args.dtype))
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-12))
likelihood.noise = np.exp(-5)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
for name, param in likelihood.named_parameters():
    if param.requires_grad:
        print(name, param.data)
print(Z.shape)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.001)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_dataset.inputs.shape[0])

from tqdm import tqdm

data_iter = iter(train_loader)

epochs_iter = tqdm(range(args.iterations), desc="Iteration")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    try:
        inputs, target = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(train_loader)
        inputs, target = next(data_iter)
    optimizer.zero_grad()
    output = model(inputs.to(args.dtype))

    loss = -mll(output, target.T)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
        
model.eval()
likelihood.eval()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
for name, param in likelihood.named_parameters():
    if param.requires_grad:
        print(name, param.data)
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))



x = train_dataset.inputs
y = train_dataset.targets * train_dataset.targets_std + train_dataset.targets_mean
plt.scatter(train_dataset.inputs, y, label="Training dataset", s=5)




preds = model(torch.tensor(test_dataset.inputs).to(args.dtype))
mean = preds.mean
var = preds.variance

plt.plot(test_dataset.inputs.flatten(), mean.detach().numpy().flatten())
plt.fill_between(test_dataset.inputs.flatten(),
                 mean.detach().numpy().flatten() - 3*var.detach().numpy().flatten(),
                 mean.detach().numpy().flatten() + 3*var.detach().numpy().flatten(),
                 alpha = 0.2
                 )
plt.show()
