import torch
import numpy as np
import time
from tqdm import tqdm
from utils import *


def train(model,
          training_generator,
          optimizer,
          val_generator=None,
          scheduler=None,
          epochs=2000,
          early_stopping=False,
          device=None):

    model.train()

    batch_size = training_generator.batch_size

    length_dataset = len(training_generator.dataset)
    if batch_size <= length_dataset:
        batch_size = length_dataset

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

                loss = model.train_step(optimizer, data.to(device),
                                        target.to(device))
                avg_nelbo += loss
                avg_rmse += model.likelihood.rmse_val
                avg_nll += model.likelihood.nll_val

            NELBO = avg_nelbo / train_per_epoch
            NLL = avg_nll / train_per_epoch
            RMSE = avg_rmse / train_per_epoch

            if val_generator is None:
                if epoch % miniters == 0:
                    tepoch.set_postfix({
                        "nelbo_train":
                        "{0:.2f}".format(NELBO.detach().cpu().numpy()),
                        "rmse_train":
                        "{0:.2f}".format(RMSE.detach().cpu().numpy()),
                        "nll_train":
                        "{0:.2f}".format(NLL.detach().cpu().numpy()),
                    })

            if val_generator is not None:
                avg_nelbo_val = 0.0
                avg_rmse_val = 0
                avg_nll_val = 0

                for data, target in val_generator:

                    loss = model.test_step(data.to(device), target.to(device))
                    avg_nelbo_val += loss
                    avg_rmse_val += model.likelihood.rmse_val
                    avg_nll_val += model.likelihood.nll_val

                NELBO_val = avg_nelbo_val / len(val_generator)
                NLL_val = avg_nll_val / len(val_generator)
                RMSE_val = avg_rmse_val / len(val_generator)
                if epoch % miniters == 0:
                    tepoch.set_postfix({
                        "nelbo_train":
                        "{0:5.2f}".format(NELBO.detach().cpu().numpy()),
                        "rmse_train":
                        "{0:5.2f}".format(RMSE.detach().cpu().numpy()),
                        "nll_train":
                        "{0:5.2f}".format(NLL.detach().cpu().numpy()),
                        "rmse_val":
                        "{0:5.2f}".format(RMSE_val.detach().cpu().numpy()),
                        "nll_val":
                        "{0:5.2f}".format(NLL_val.detach().cpu().numpy()),
                    })

            if scheduler is not None:
                scheduler.step()

    return {"nelbo": NELBO, "nll": NLL, "rmse": RMSE}


def test(model, generator, device=None):

    model.train()

    batch_size = generator.batch_size

    length_dataset = len(generator.dataset)
    if batch_size <= length_dataset:
        batch_size = length_dataset

    avg_nelbo = 0.0
    avg_rmse = 0
    avg_nll = 0

    for data, target in generator:

        loss = model.test_step(data.to(device), target.to(device))
        avg_nelbo += loss
        avg_rmse += model.likelihood.rmse_val
        avg_nll += model.likelihood.nll_val

    NELBO = avg_nelbo / len(generator)
    NLL = avg_nll / len(generator)
    RMSE = avg_rmse / len(generator)

    return {
        "nelbo": NELBO.detach().cpu().numpy(),
        "nll": NLL.detach().cpu().numpy(),
        "rmse": RMSE.detach().cpu().numpy()
    }


def predict(model, generator, device=None):

    model.eval()

    batch_size = generator.batch_size

    length_dataset = len(generator.dataset)
    if batch_size <= length_dataset:
        batch_size = length_dataset

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

    model.eval()

    batch_size = generator.batch_size
    length_dataset = len(generator.dataset)

    if batch_size <= length_dataset:
        batch_size = length_dataset

    # Generate variables and operations for the minimizer and initialize variables
    train_per_epoch = len(generator)
    prior = []
    for idx, data in enumerate(generator):
        try:
            batch_x, _ = data
        except:
            batch_x = data
        prior_samples = model.get_prior_samples(batch_x.to(device))

        prior.append(prior_samples.detach().numpy())

    prior = np.concatenate(prior, axis=2)

    return prior
