from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from src.dvip import DVIP_Base, IVAE
from src.layers_init import init_layers_vae

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit, score
from scripts.filename import create_file_name

args = manage_experiment_configuration()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args.device)
torch.manual_seed(args.seed)

train_dataset, train_test_dataset, test_dataset = args.dataset.get_split(
    args.test_size, args.seed + args.split
)

# Get VIP layers
layers = init_layers_vae(train_dataset.inputs, args.dataset.output_dim, **vars(args))

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_test_loader = DataLoader(train_test_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Create DVIP object
dvip = IVAE(
    args.likelihood,
    layers[0],
    layers[1],
    len(train_dataset),
    bb_alpha=args.bb_alpha,
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
    device=args.device,
)
dvip.print_variables()

# Define optimizer and compile model
opt = torch.optim.Adam(dvip.parameters(), lr=args.lr)

# Set the number of training samples to generate
dvip.num_samples = args.num_samples_train
# Train the model
start = timer()
loss = fit(
    dvip,
    train_loader,
    opt,
    use_tqdm=True,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
)
end = timer()


dvip.print_variables()

import matplotlib.pyplot as plt

plt.plot(loss[(2 * len(loss) // 3) :])
plt.savefig("plots/ev_" + create_file_name(args) + ".pdf", format="pdf")


n = 10
imgs = train_dataset.inputs[0:n]
f, axis = plt.subplots(n // 2, 4)
# f.tight_layout()  # Or equivalently,  "plt.tight_layout()"
for i in range(n):
    axis[i % (n // 2)][2 * (i // (n // 2))].imshow(imgs[i].reshape(28, 28))
    # plt.show()
    pred = dvip.predict_y(
        torch.tensor(imgs[i], dtype=args.dtype, device=args.device).unsqueeze(0), 1
    )
    pred = pred[0].cpu().reshape(28, 28).detach().numpy()
    # pred = 1 / (1 + np.exp(-pred))
    axis[i % (n // 2)][2 * (i // (n // 2)) + 1].imshow(pred)


plt.savefig("plots/" + create_file_name(args) + ".pdf", format="pdf")

# Set the number of test samples to generate
dvip.num_samples = args.num_samples_test

# Test the model
# train_metrics = score(
#     dvip, train_test_loader, args.metrics, use_tqdm=True, device=args.device
# )
test_metrics = score(dvip, test_loader, args.metrics, use_tqdm=True, device=args.device)

# print("TRAIN RESULTS: ")
# for k, v in train_metrics.items():
#     print("\t - {}: {}".format(k, v))

print("TEST RESULTS: ")
for k, v in test_metrics.items():
    print("\t - {}: {}".format(k, v))


d = {
    **vars(args),
    **{"time": end - start},
    # **{k + "_train": v for k, v in train_metrics.items()},
    **test_metrics,
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df.to_csv(
    path_or_buf="results/" + create_file_name(args) + ".csv",
    encoding="utf-8",
)
