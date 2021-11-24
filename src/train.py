import torch
import numpy as np
import time
from tqdm import tqdm
from utils import *
import pkbar


def train(model, training_generator, optimizer, epochs=2000, device=None):

    model.train()

    batch_size = training_generator.batch_size

    length_dataset = len(training_generator.dataset)
    assert batch_size <= length_dataset

    # Generate variables and operations for the minimizer and initialize variables
    train_per_epoch = len(training_generator)

    miniters = 10
    with tqdm(range(epochs), unit="epoch", miniters=miniters) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Training ")
            avg_nelbo = 0.0
            avg_rmse = 0
            avg_nll = 0

            for data, target in training_generator:

                loss = model.train_step(
                    optimizer, data.to(device), target.to(device)
                )
                avg_nelbo += loss
                avg_rmse += model.likelihood.rmse_val
                avg_nll += model.likelihood.nll_val

                NELBO = avg_nelbo / train_per_epoch
                NLL = avg_nll / train_per_epoch
                RMSE = avg_rmse / train_per_epoch

                if epoch % miniters == 0:
                    tepoch.set_postfix(
                        {
                            "nelbo_train": "{:3f}".format(
                                NELBO.detach().cpu().numpy()
                            ),
                            "rmse_train": "{:3f}".format(
                                RMSE.detach().cpu().numpy()
                            ),
                            "nll_train": "{:3f}".format(
                                NLL.detach().cpu().numpy()
                            ),
                        }
                    )


def predict(model, generator, device=None):

    model.eval()

    batch_size = generator.batch_size

    length_dataset = len(generator.dataset)
    assert batch_size <= length_dataset

    # Generate variables and operar)
    means, vars = [], []
    for idx, data in enumerate(generator):
        try:
            batch_x, _ = data
        except:
            batch_x = data
        batch_means, batch_vars = model(batch_x.to(device))
        means.append(batch_means.detach().numpy())
        vars.append(batch_vars.detach().numpy())

    means = np.concatenate(means, axis=1)
    vars = np.concatenate(vars, axis=1)

    return means, vars


def predict_prior_samples(model, generator, device=None):

    batch_size = generator.batch_size
    length_dataset = len(generator.dataset)

    assert batch_size <= length_dataset

    # Generate variables and operations for the minimizer and initialize variables
    train_per_epoch = len(generator)
    prior = []
    for idx, data in enumerate(generator):
        try:
            batch_x, _ = data
        except:
            batch_x = data
        prior_samples = model.predict_prior_samples(batch_x.to(device))

        prior.append(prior_samples.detach().numpy())

    prior = np.concatenate(prior)

    return prior
