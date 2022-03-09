import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import pandas as pd

sys.path.append(".")

from scripts.filename import create_file_name

from src.dvip import DVIP_Base
from src.layers_init import init_layers
from utils.plotting_utils import build_plot_name, learning_curve, plot_train_test
from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, fit_with_metrics, predict, predict_prior_samples, score

args = manage_experiment_configuration()

torch.manual_seed(args.seed)

train_d, train_test_d, test_d = args.dataset.get_split(split = args.split)


# Get VIP layers
layers = init_layers(train_d.inputs, args.dataset.output_dim, **vars(args))

train_loader = DataLoader(train_d, batch_size=args.batch_size, shuffle = True)
train_test_loader = DataLoader(train_test_d, batch_size=args.batch_size)
test_loader = DataLoader(test_d, batch_size=args.batch_size)


dvip = DVIP_Base(
    args.likelihood,
    layers,
    len(train_d),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_d.targets_mean,
    y_std=train_d.targets_std,
    dtype=args.dtype,
    device=args.device,
)

dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)


# Train the model
train_hist, val_hist = fit_with_metrics(
    dvip,
    train_loader,
    opt,
    args.metrics,
    val_generator=test_loader,
    epochs=args.epochs,
    device=args.device,
)

dvip.print_variables()

def get_predictive_results(mean, var):

    prediction_mean = np.mean(mean, axis=0)
    prediction_var = np.mean(var + mean ** 2, axis=0) - prediction_mean ** 2
    return prediction_mean, prediction_var


# Change MC samples for test
dvip.num_samples = args.num_samples_test
train_metrics = score(dvip, train_test_loader, args.metrics, device=args.device)
test_metrics = score(dvip, test_loader, args.metrics, device=args.device)

test_metrics_names = list(test_metrics.keys())
num_metrics = len(test_metrics_names)

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))
    
    
test_mean, test_var = predict(dvip, train_test_loader, device=args.device)
test_prediction_mean, test_prediction_var = get_predictive_results(test_mean, test_var)

dvip.eval()
prior_samples = predict_prior_samples(dvip, train_test_loader).T[0, :, :, -1]

# Create plot title and path
fig_title, path = build_plot_name(**vars(args))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 9))

plt.scatter(train_d.inputs * train_d.inputs_std + train_d.inputs_mean, 
            train_d.targets* train_d.targets_std + train_d.targets_mean,
            s = 0.2,
            label = "Training set")
plt.scatter(test_d.inputs * train_d.inputs_std + train_d.inputs_mean, 
            test_d.targets,
            s = 0.2,
            color = "purple",
            label = "Test set")

X = train_test_d.inputs* train_d.inputs_std + train_d.inputs_mean
plt.plot(X,
         test_prediction_mean,
         color = "orange",
         label = "Predictive mean")
plt.fill_between(X.flatten(),
                 (test_prediction_mean - 2*np.sqrt(test_prediction_var)).flatten(),
                 (test_prediction_mean + 2*np.sqrt(test_prediction_var)).flatten(),
                 color = "orange",
                 alpha = 0.3,
                 label = "Predictive std")

ymin, ymax = plt.ylim()

plt.plot(X.flatten(), prior_samples[:, :-1], color = "red", alpha = 0.05)
plt.plot(X.flatten(), prior_samples[:, -1], color = "red", alpha = 0.05, label = "Prior samples")
plt.ylim([ymin, ymax])
plt.legend()

plt.savefig("plots/extrapolate_" + create_file_name(args) + ".pdf")


test_metrics_names = list(test_metrics.keys())
num_metrics = len(test_metrics_names)

df = pd.DataFrame.from_dict(train_hist)
df_val = pd.DataFrame.from_dict(val_hist)

learning_curve(df, df_val, test_metrics_names, num_metrics, args)

d = {
    **vars(args),
    **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df.to_csv(
    path_or_buf="results/extrapolation_" + create_file_name(args) + ".csv",
    encoding="utf-8",
)
