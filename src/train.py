import torch
import numpy as np
import time

from utils import *
import pkbar


def train(model, training_generator, optimizer, epochs=2000, device=None):

    model.train()

    batch_size = training_generator.batch_size

    length_dataset = len(training_generator.dataset)
    assert batch_size <= length_dataset

    # Generate variables and operations for the minimizer and initialize variables
    global_nelbow = np.inf
    train_per_epoch = len(training_generator)
    print(train_per_epoch)

    initial_time = time.time()
    total_time = 0.0
    for epoch in range(epochs):
        message_write = {}
        message_write['epoch'] = epoch
        kbar = pkbar.Kbar(target=train_per_epoch,
                          epoch=epoch,
                          num_epochs=epochs,
                          width=50,
                          always_stateful=False)
        avg_nelbo = 0.0
        avg_rmse = 0
        avg_nll = 0

        kbar.update(0, values=[("nelbo", 0), ("rmse", 0), ("nll", 0)])
        for idx, data in enumerate(training_generator):
            batch_x, batch_y = data
            loss = model.train_step(optimizer, batch_x.to(device),
                                    batch_y.to(device))

            avg_nelbo += loss
            rmse = model.likelihood.rmse_val
            nll = model.likelihood.nll_val
            avg_rmse += rmse
            avg_nll += nll
            kbar.update(idx + 1,
                        values=[("nelbo", loss), ("rmse", rmse), ("nll", nll)])

        NELBO = avg_nelbo / train_per_epoch
        NLL = avg_nll / train_per_epoch
        ACC = avg_rmse / train_per_epoch

        message_write['acc_train'] = ACC.detach().cpu().numpy()
        message_write['nelbo_train'] = NELBO.detach().cpu().numpy()
        message_write['nll_train'] = NLL.detach().cpu().numpy()

    total_time = time.time() - initial_time

    return total_time


def predict(model, generator, device=None):

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
        batch_means, batch_vars = model.predict(batch_x.to(device))
        means.append(batch_means.detach().numpy())
        vars.append(batch_vars.detach().numpy())

    means = np.concatenate(means)
    vars = np.concatenate(vars)

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
