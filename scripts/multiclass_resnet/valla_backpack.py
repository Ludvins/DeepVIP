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
from utils.pytorch_learning import fit_map_crossentropy, fit, forward, score
from scripts.filename import create_file_name
from src.valla import VaLLAMultiClassBackPack
from utils.models import get_resnet
from utils.dataset import get_dataset
from utils.metrics import SoftmaxClassification, OOD
from src.utils import smooth
import matplotlib.pyplot as plt
args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)

train_dataset, val_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

f = get_resnet("resnet20", 10)

Z = kmeans2(train_dataset.inputs.reshape(train_dataset.inputs.shape[0], -1), 
            args.num_inducing, minit="points", seed=args.seed)[0]
Z = Z.reshape(args.num_inducing, *train_dataset.inputs.shape[1:])

valla = VaLLAMultiClassBackPack(
    f,
    Z,
    prior_std=args.prior_std,
    num_data=train_dataset.inputs.shape[0],
    output_dim=train_dataset.output_dim,
    track_inducing_locations=True,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    alpha = args.bb_alpha,
    device=args.device,
    dtype=args.dtype,
    #seed = args.seed
)

opt = torch.optim.Adam(valla.parameters(recurse=False), lr=args.lr)

start = timer()
loss = fit(
    valla,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=args.verbose,
    iterations=args.iterations,
    device=args.device,
)
end = timer()


save_str = "VaLLA_Conv_dataset={}_M={}_seed={}".format(
    args.dataset_name, args.num_inducing, args.seed
)


plt.savefig("plots/" + save_str + ".pdf")

test_metrics = score(
    valla,
    test_loader,
    SoftmaxClassification,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
)

test_metrics["prior_std"] = valla.prior_std.detach().numpy()
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["seed"] = args.seed
test_metrics["alpha"] = args.bb_alpha
test_metrics["timer"] = end-start

if args.test_ood:
    ood_dataset = args.dataset.get_ood_datasets()
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size)
    ood_metrics = score(
        valla, ood_loader, OOD, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
    )
    test_metrics["OOD-AUC"] = ood_metrics["AUC"]
    test_metrics["OOD-AUC MC"] = ood_metrics["AUC MC"]

if args.test_corruptions:
    for corruption_value in args.dataset.corruption_values:
        corrupted_dataset = args.dataset.get_corrupted_split(corruption_value)

        loader = DataLoader(corrupted_dataset, batch_size=args.batch_size)
        corrupted_metrics = score(
            valla, loader, SoftmaxClassification, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
        ).copy()
        print(corrupted_metrics)

        test_metrics = {
            **test_metrics,
            **{k+'-C'+str(corruption_value): v for k, v in corrupted_metrics.items()}
        }



df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
