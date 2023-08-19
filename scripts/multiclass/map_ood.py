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
from utils.pytorch_learning import fit_map_crossentropy, acc_multiclass, fit, forward, score
from scripts.filename import create_file_name
from src.generative_functions import *
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.dataset import get_dataset
from utils.metrics import SoftmaxClassification, OOD

from tqdm import tqdm
from src.utils import smooth

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

args.dataset = get_dataset(args.dataset_name)
train_dataset, test_dataset, ood_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size)



f = get_mlp(
    train_dataset.inputs.shape[1],
    args.dataset.output_dim,
    [200, 200],
    torch.nn.Tanh,
    device=args.device,
    dtype=args.dtype,
)
if args.weight_decay != 0:
    args.prior_std = np.sqrt(1 / (len(train_dataset) * args.weight_decay))


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)

# Set the number of training samples to generate



try:
    f.load_state_dict(torch.load("weights/multiclass_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map_crossentropy(
        f,
        train_loader,
        opt,
        criterion=torch.nn.CrossEntropyLoss(),
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
    )
    import matplotlib.pyplot as plt
    iters_per_epoch = len(train_loader)
    if len(loss) > iters_per_epoch:
        loss = smooth(np.array(loss), iters_per_epoch)

        plt.plot(loss)
        plt.show()
    end = timer()
    train_acc = acc_multiclass(
        f, train_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    test_acc = acc_multiclass(
        f, test_loader, use_tqdm=True, device=args.device, dtype=args.dtype
    )

    print("Train acc: ", train_acc)
    print("Test acc: ", test_acc)
    torch.save(f.state_dict(), "weights/multiclass_weights_" + args.dataset_name)

save_str = "MAP_dataset={}".format(
    args.dataset_name)


y_mean = torch.tensor(train_dataset.targets_mean, device=args.device)
y_std = torch.tensor(train_dataset.targets_std, device=args.device)


def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
    if y.ndim == 1:
        y = y.unsqueeze(-1)

        # Cast types if needed.
    if args.dtype != X.dtype:
        X = X.to(args.dtype)
    if args.dtype != y.dtype:
        y = y.to(args.dtype)

    Fmean = f(X)  # Forward pass
    Fvar = torch.zeros(Fmean.shape[0], Fmean.shape[1], Fmean.shape[1],
                       dtype = args.dtype, device = args.device)

    return 0, Fmean * y_std + y_mean, Fvar * y_std**2

f.test_step = test_step

test_metrics = score(
    f,
    test_loader,
    SoftmaxClassification,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)

ood_metrics = score(
    f, ood_loader, OOD, use_tqdm=True, device=args.device, dtype=args.dtype
)

test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations

test_metrics["OOD-AUC"] = ood_metrics["AUC"]
test_metrics["OOD-AUC MC"] = ood_metrics["AUC MC"]


df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
