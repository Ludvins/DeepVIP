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
from src.sparseLA import SparseLA, GPLLA
from src.likelihood import Gaussian
from src.backpack_interface import BackPackInterface
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




f = get_mlp(train_dataset.inputs.shape[1], 
            train_dataset.targets.shape[1], 
            [50, 50], 
            torch.nn.Tanh, device = args.device,dtype =  args.dtype)

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


Z = kmeans2(train_dataset.inputs, args.num_inducing, minit="points", seed=args.seed)[0]

sparseLA = SparseLA(
    f.forward,
    Z, 
    alpha = args.bb_alpha,
    prior_std=2.2026465,
    likelihood=Gaussian(device=args.device, 
                        log_variance=np.log(0.11802231**2),
                        dtype = args.dtype), 
    num_data = train_dataset.inputs.shape[0],
    output_dim = 1,
    backend = BackPackInterface(f, "regression"),
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



lla = GPLLA(f, 
            prior_std = 2.2026465,
            likelihood_hessian=lambda x,y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0) / 0.11802231**2,
            likelihood=Gaussian(device=args.device, 
                        log_variance=np.log(0.11802231**2),
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            device = args.device,
            dtype = args.dtype)


lla.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
        torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))



import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
fig, axis = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[2,1]}, figsize=(16, 10))


axis[0].scatter(train_dataset.inputs, train_dataset.targets, label = "Training points")



_, valla_var = predict(sparseLA, test_loader)
_, lla_var = predict(lla, test_loader)

valla_std = np.sqrt(valla_var)
lla_std = np.sqrt(lla_var)

KL = - np.log(lla_std) + np.log(valla_std) - 0.5 + ((lla_std**2)/(2*valla_std**2))
KL = np.sum(KL)

MAE = np.mean(np.abs(valla_std - lla_std))

print("KL:", KL)
print("MAE:", MAE)


d = {
    "M": args.num_inducing,
    "seed": args.seed,
    "KL": KL,
    "MAE": MAE
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/VaLLA_dataset={}_M={}_seed={}.csv".format(args.dataset_name,args.num_inducing, args.seed),
    encoding="utf-8",
)