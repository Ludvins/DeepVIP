from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

from laplace.curvature import BackPackInterface

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, predict
from utils.models import get_mlp
from utils.dataset import get_dataset
from src.likelihood import Gaussian
from src.sparseLA import GPLLA, ELLA

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



f = get_mlp(train_dataset.inputs.shape[1], train_dataset.targets.shape[1], [50, 50], torch.nn.Tanh, device = args.device, dtype = args.dtype)


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





ella = ELLA(f, 
            args.num_inducing,
            np.min([args.num_inducing, 20]),
            prior_std = 2.2026465,
            likelihood_hessian=lambda x,y: torch.ones_like(y).unsqueeze(-1).permute(1, 2, 0) / 0.11802231**2,
            likelihood=Gaussian(device=args.device, 
                        log_variance=np.log(0.11802231**2),
                        dtype = args.dtype), 
            backend = BackPackInterface(f, "classification"),
            seed = args.seed,
            device = args.device,
            dtype = args.dtype)



ella.fit(torch.tensor(train_dataset.inputs, device = args.device, dtype = args.dtype),
        torch.tensor(train_dataset.targets, device = args.device, dtype = args.dtype))


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



_, ella_var = predict(ella, test_loader)
_, lla_var = predict(lla, test_loader)

ella_std = np.sqrt(ella_var)
lla_std = np.sqrt(lla_var)

KL = - np.log(lla_std) + np.log(ella_std) - 0.5 + ((lla_std**2)/(2*ella_std**2))
KL = np.sum(KL)

MAE = np.mean(np.abs(ella_std - lla_std))

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
    path_or_buf="results/ELLA_dataset={}_M={}_seed={}.csv".format(args.dataset_name,args.num_inducing, args.seed),
    encoding="utf-8",
)