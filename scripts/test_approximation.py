from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from scipy.cluster.vq import kmeans2
from functorch import jacrev, hessian, jacfwd

sys.path.append(".")

from src.fvi import FVI
from src.layers_init import init_layers, init_layers_tvip
from src.layers import TVIPLayer
from src.likelihood import QuadratureGaussian
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score, predict
from scripts.filename import create_file_name
from src.generative_functions import *

args = manage_experiment_configuration()

# args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

Z = kmeans2(train_dataset.inputs, 10, minit="points", seed=args.seed)[0]

f1 = BayesianNN(
    input_dim=1,
    structure=args.bnn_structure,
    activation=args.activation,
    output_dim=1,
    layer_model=BayesLinear,
    dropout=args.dropout,
    fix_mean = args.fix_mean,
    device=args.device,
    seed=args.seed,
    dtype=args.dtype,
)



# f1 = GP(
#     input_dim=1,
#     output_dim=1,
#     inner_layer_dim=20,
#     device=args.device,
#     seed=args.seed,
#     dtype=args.dtype,
# )


X = test_dataset.inputs
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 12))
ax1 = plt.subplot(4, 1, 1)
ax2 = plt.subplot(4, 2, 3)
ax3 = plt.subplot(4, 2, 4)
ax4 = plt.subplot(4, 2, 5)
ax5 = plt.subplot(4, 2, 6)
ax6 = plt.subplot(4, 2, 7)
ax7 = plt.subplot(4, 2, 8)
fig = plt.gcf()




w = f1.get_weights()

J = jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X), w)
J = torch.cat([torch.flatten(a, -2, -1) for a in J], dim=-1).transpose(-1, -2)


H = hessian(f1.forward_weights, argnums = 1)(torch.tensor(X), w)


H = torch.cat([
        torch.cat([
            H[i][j].flatten(-4, -3).flatten(-2, -1) 
        for j in range(len(H))], dim=-1)
    for i in range(len(H))], dim = -2)


#H = torch.diagonal(H, dim1=-2, dim2 = -1).transpose(-1, -2)


S = torch.exp(f1.get_std_params()) ** 2

mean_weights = f1.get_weights()
weights = f1.sample_weights(200)


samples = f1.forward_weights(torch.tensor(X, dtype=args.dtype), weights)
for f in samples:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")



diff = [weights[i] - mean_weights[i] for i in range(len(weights))]

diff = torch.cat([torch.flatten(a, -2, -1) for a in diff], -1)


samples2 = f1.forward_mean(torch.tensor(X)).unsqueeze(0)# + torch.einsum("nsd, ms -> mnd", J, diff) + 0.5 * torch.einsum("ma, ndba, mb -> mnd",diff, H, diff)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax2.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)
ax2.set_title("Samples from Taylor Approximation of Order 0")
ax3.set_title("Mean Absolute Error of Taylor Approximation of Order 0")

ax3.plot(X.flatten(), err0.flatten())


samples2 = f1.forward_mean(torch.tensor(X)) + torch.einsum("nsd, ms -> mnd", J, diff)# + 0.5 * torch.einsum("ma, ndba, mb -> mnd",diff, H, diff)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax4.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax5.plot(X.flatten(), err0.flatten())
ax4.set_title("Samples from Taylor Approximation of Order 1")
ax5.set_title("Mean Absolute Error of Taylor Approximation of Order 1")

samples2 = f1.forward_mean(torch.tensor(X)) + torch.einsum("nsd, ms -> mnd", J, diff) + 0.5 * torch.einsum("ma, ndba, mb -> mnd",diff, H, diff)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax6.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax7.plot(X.flatten(), err0.flatten())
ax6.set_title("Samples from Taylor Approximation of Order 2")
ax7.set_title("Mean Absolute Error of Taylor Approximation of Order 2")
plt.show()
