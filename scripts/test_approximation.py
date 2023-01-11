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

ax1 = plt.subplot(7, 1, 1)
ax2 = plt.subplot(7, 2, 3)
ax3 = plt.subplot(7, 2, 4)
ax4 = plt.subplot(7, 2, 5)
ax5 = plt.subplot(7, 2, 6)
ax6 = plt.subplot(7, 2, 7)
ax7 = plt.subplot(7, 2, 8)
ax8 = plt.subplot(7, 2, 9)
ax9 = plt.subplot(7, 2, 10)
ax10 = plt.subplot(7, 2, 11)
ax11 = plt.subplot(7, 2, 12)
ax12 = plt.subplot(7, 2, 13)
ax13 = plt.subplot(7, 2, 14)

fig = plt.gcf()




w = f1.get_weights()

J = jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X), w)
print(J.shape)

S = torch.exp(f1.get_std_params()) ** 2

mean_weights = w
weights = f1.sample_weights(200)


samples = f1.forward_weights(torch.tensor(X, dtype=args.dtype), weights)
for f in samples:
    ax1.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax1.set_title("Samples from original IP")



diff = weights - mean_weights
print(diff.shape)

samples2 = f1.forward_mean(torch.tensor(X)).unsqueeze(0)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax2.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)
ax2.set_title("Samples from Taylor Approximation of Order 0")
ax3.set_title("Mean Absolute Error of Taylor Approximation of Order 0")

ax3.plot(X.flatten(), err0.flatten())


samples2 = f1.forward_mean(torch.tensor(X)) + torch.einsum("nds, ms -> mnd", J, diff)
print(samples2.shape)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax4.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax5.plot(X.flatten(), err0.flatten())
ax4.set_title("Samples from Taylor Approximation of Order 1")
ax5.set_title("Mean Absolute Error of Taylor Approximation of Order 1")


# H = hessian(f1.forward_weights, argnums = 1)(torch.tensor(X), w)
# H = torch.diagonal(H, dim1 = -2, dim2 = -1)
# print(H.shape)
# input()
# samples2 = f1.forward_mean(torch.tensor(X)) + torch.einsum("nds, ms -> mnd", J, diff)
# samples2 = samples2 +  torch.einsum("nds, ms -> mnd", H, diff**2)
# err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

# for f in samples2:
#     ax6.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

# ax7.plot(X.flatten(), err0.flatten())
# ax6.set_title("Samples from Taylor Approximation of Order 2")
# ax7.set_title("Mean Absolute Error of Taylor Approximation of Order 2")



n = 10
weights_ensemble = f1.sample_weights(n)
J = torch.stack(
    [jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), a) for a in weights_ensemble]
    )

diff = weights.unsqueeze(1) - weights_ensemble.unsqueeze(0)

samples2 = f1.forward_weights(torch.tensor(X), weights_ensemble) \
    + torch.einsum("ands, bas -> band", J, diff)

samples2 = torch.mean(samples2, axis = 1)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax6.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax7.plot(X.flatten(), err0.flatten())
ax6.set_title("Samples from Taylor Approximation Ensemble {}".format(n))
ax7.set_title("Mean Absolute Error of Taylor Approximation Ensemble {}".format(n))

n = 20
weights_ensemble = f1.sample_weights(n)
J = torch.stack(
    [jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), a) for a in weights_ensemble]
    )

diff = weights.unsqueeze(1) - weights_ensemble.unsqueeze(0)

samples2 = f1.forward_weights(torch.tensor(X), weights_ensemble) \
    + torch.einsum("ands, bas -> band", J, diff)

samples2 = torch.mean(samples2, axis = 1)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax8.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax9.plot(X.flatten(), err0.flatten())
ax8.set_title("Samples from Taylor Approximation Ensemble {}".format(n))
ax9.set_title("Mean Absolute Error of Taylor Approximation Ensemble {}".format(n))

n = 50
weights_ensemble = f1.sample_weights(n)
J = torch.stack(
    [jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), a) for a in weights_ensemble]
    )

diff = weights.unsqueeze(1) - weights_ensemble.unsqueeze(0)

samples2 = f1.forward_weights(torch.tensor(X), weights_ensemble) \
    + torch.einsum("ands, bas -> band", J, diff)

samples2 = torch.mean(samples2, axis = 1)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax10.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax11.plot(X.flatten(), err0.flatten())
ax10.set_title("Samples from Taylor Approximation Ensemble {}".format(n))
ax11.set_title("Mean Absolute Error of Taylor Approximation Ensemble {}".format(n))




n = 100
weights_ensemble = f1.sample_weights(n)
J = torch.stack(
    [jacrev(f1.forward_weights, argnums = 1)(torch.tensor(X, dtype=args.dtype), a) for a in weights_ensemble]
    )

diff = weights.unsqueeze(1) - weights_ensemble.unsqueeze(0)

samples2 = f1.forward_weights(torch.tensor(X), weights_ensemble) \
    + torch.einsum("ands, bas -> band", J, diff)

samples2 = torch.mean(samples2, axis = 1)
err0 = np.mean(np.abs(samples.detach().cpu().numpy() - samples2.detach().cpu().numpy()), axis = 0)

for f in samples2:
    ax12.plot(X.flatten(), f.detach().numpy().flatten(), alpha=0.5)

ax13.plot(X.flatten(), err0.flatten())
ax12.set_title("Samples from Taylor Approximation Ensemble {}".format(n))
ax13.set_title("Mean Absolute Error of Taylor Approximation Ensemble {}".format(n))







plt.tight_layout()
plt.savefig("MAE.pdf", format = "pdf")
plt.show()
