import numpy as np
import torch
from .metrics import Metrics
from tqdm import tqdm


def fit(
    model,
    training_generator,
    optimizer,
    scheduler=None,
    epochs=2000,
    device=None,
):
    """
    Trains the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    training_generator : iterable
                         Must return batches of pairs corresponding to the
                         given inputs and target values.
    optimizer : torch optimizer
                The already initialized optimizer.
    scheduler : torch scheduler
                Learning rate scheduler.
    epochs : int
             Number of epochs to train de model.
    device : torch device
             Device in which to perform all computations.
    """
    # Set model in training mode
    model.train()
    for _ in range(epochs):
        # Mini-batch training
        for inputs, target in training_generator:
            inputs = inputs.to(device)
            target = target.to(device)
            model.train_step(optimizer, inputs, target)

        # Update learning rate using scheduler if available
        if scheduler is not None:
            scheduler.step()


def score(model, generator, device=None):
    """
    Evaluates the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    generator : iterable
                Must return batches of pairs corresponding to the
                given inputs and target values.
    device : torch device
             Device in which to perform all computations.

    Returns
    -------
    metrics : dictionary
              Contains pairs of (metric, value) averaged over the number of
              batches.
    """
    # Set model in evaluation mode
    model.eval()
    # Initialize metrics
    metrics = Metrics(len(generator.dataset), device=device)
    with torch.no_grad():
        # Batches evaluation
        for data, target in generator:
            data = data.to(device)
            target = target.to(device)
            loss, mean_pred, std_pred = model.test_step(data, target)
            # Update mertics using this batch
            metrics.update(target, loss, mean_pred, std_pred, light=False)
    # Return metrics as a dictionary
    return metrics.get_dict()


def predict(model, generator, device=None):

    with torch.no_grad():
        # Generate variables and operar)
        means, vars = [], []
        for idx, data in enumerate(generator):
            try:
                batch_x, _ = data
            except:
                batch_x = data
            batch_means, batch_vars = model(batch_x.to(device))
            means.append(batch_means.detach().cpu().numpy())
            vars.append(batch_vars.detach().cpu().numpy())

        means = np.concatenate(means, axis=1)
        vars = np.concatenate(vars, axis=1)

    return means, vars


def fit_with_metrics(
    model,
    training_generator,
    optimizer,
    val_generator=None,
    scheduler=None,
    epochs=2000,
    device=None,
    verbose=1,
):

    # Array storing metrics during training
    metrics = Metrics(len(training_generator.dataset), device=device)
    if val_generator is not None:
        metrics_val = Metrics(len(val_generator.dataset), device=device)

    history = []
    history_val = []

    # TQDM update interval
    miniters = 10

    if verbose == 1:
        # initialize TQDM bar
        tepoch = tqdm(range(epochs), unit="epoch", miniters=miniters)
        tepoch.set_description("Training ")
    else:
        tepoch = range(epochs)

    for epoch in tepoch:
        # Mini-batch training
        model.train()
        for data, target in training_generator:
            # Compute loss value
            data = data.to(device)
            target = target.to(device)
            loss = model.train_step(optimizer, data, target)
            # Update metrics for the given batch
            with torch.no_grad():
                mean_pred, std_pred = model(data)
                metrics.update(
                    target * model.y_std + model.y_mean,
                    loss,
                    mean_pred,
                    std_pred,
                )

        # Store history of metrics
        metrics_dict = metrics.get_dict()
        history.append(metrics_dict)
        # Reset current metrics for next epochs or validation
        metrics.reset()

        val_postfix = {}
        if val_generator is not None:
            model.eval()
            with torch.no_grad():
                for data, target in val_generator:
                    data = data.to(device)
                    target = target.to(device)
                    loss, mean_pred, std_pred = model.test_step(data, target)
                    metrics_val.update(target, loss, mean_pred, std_pred)

            metrics_val_dict = metrics_val.get_dict()
            metrics_val.reset()
            history_val.append(metrics_val_dict)

            # Handle Validation metrics in TQDM
            if verbose == 1:
                val_postfix = {
                    "rmse_val": "{0:.2f}".format(metrics_val_dict["RMSE"]),
                    "nll_val": "{0:.2f}".format(metrics_val_dict["NLL"]),
                }

        # Show metrics in TQDM
        if verbose == 1 and epoch % miniters == 0:
            tepoch.set_postfix(
                {
                    **{
                        "loss_train": "{0:.2f}".format(metrics_dict["LOSS"]),
                        "rmse_train": "{0:.2f}".format(metrics_dict["RMSE"]),
                        "nll_train": "{0:.2f}".format(metrics_dict["NLL"]),
                    },
                    **val_postfix,
                }
            )

        if scheduler is not None:
            scheduler.step()

    if val_generator is None:
        return history
    return history, history_val


def predict_prior_samples(model, generator, device=None):

    with torch.no_grad():
        # Generate variables and operations for the minimizer and initialize variables
        prior = []
        for data in generator:
            try:
                batch_x, _ = data
            except:
                batch_x = data
            prior_samples = model.get_prior_samples(batch_x.to(device))

            prior.append(prior_samples.detach().numpy())

        prior = np.concatenate(prior, axis=2)

    return prior
