import numpy as np
import torch
from .metrics import MetricsRegression, MetricsClassification
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


def score(model, generator, metrics, device=None):
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
    metrics = metrics(len(generator.dataset), device=device)
    with torch.no_grad():
        # Batches evaluation
        for data, target in generator:
            data = data.to(device)
            target = target.to(device)
            loss, mean_pred, std_pred = model.test_step(data, target)
            #log_likelihood = model.predict_logdensity(data, target)
            # Update mertics using this batch
            metrics.update(target, loss, mean_pred, std_pred, 
                    model.likelihood, light=False)
    # Return metrics as a dictionary
    return metrics.get_dict()


def predict(model, generator, device=None):
    """
    Creates the model predictions for the given data generator.
    The model predictive distribution is a Gaussian mixture.

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
    means : numpy darray of shape (S, N, D)
            Contains the predicted mean for every mixture (S),
            datapoint (N) with the corresponding output dimension (D).
    vars : numpy darray of shape (S, N, D)
           Contains the predicted variance of every mixture (S),
           datapoint (N), with the corresponding output dimension (D).
    """

    # Sets the model in evaluation mode. Dropout layers are not used.
    model.eval()
    # Speed-up predictions telling torch not to compute gradients.
    with torch.no_grad():
        # Create containers
        means, sqrts = [], []

        # Look over batches of data.
        for idx, data in enumerate(generator):
            # Consider datasets with no targets, just input values.
            try:
                batch_x, _ = data
            except:
                batch_x = data
            # Create batched predictions
            batch_means, batch_vars = model(batch_x.to(device))
            # Apped to the arrays
            means.append(batch_means.detach().cpu().numpy())
            sqrts.append(batch_vars.detach().cpu().numpy())

        # Concatenate batches on second dimension, the first
        # corresponds to the mixture.
        means = np.concatenate(means, axis=1)
        sqrts = np.concatenate(sqrts, axis=1)

    return means, sqrts


def fit_with_metrics(
    model,
    training_generator,
    optimizer,
    metric,
    val_generator=None,
    val_mc_samples=20,
    scheduler=None,
    epochs=2000,
    device=None,
    verbose=1,
):
    """
    Fits the given model using the training generator and optimizer.
    Returns (and shows using tqdm if verbose is 1) training and
    validation metrics.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    training_generator : iterable
                         Generates batches of training data
    optimizer : torch.optimizer
                The considered optimization optimizer.
    val_generator : iterable
                    Generates batches of validation data.
    val_mc_samples : int
                     Number of MC samples to use in validation.
    echeduler : torch scheduler
                Learning rate scheduler.
    epochs : int
             Number of training epochs, i.e, complete loops
             over the complete set of data.
    device : torch device
             Device in which to perform all computations.
    verbose : int
              If 1, tqdm shows a progress var with the eta
              and metrics.

    Returns
    -------
    history : array
              Contains dictionaries with the training metrics
              at each epoch.
    history_val : array
                  Contains dictionarus with the validation
                  metrics at each epoch. Only if val_generator
                  is not none.
    """

    # Store the number of training MC samples. This is done as
    # backup as this value is altered in validation
    train_mc_samples = model.num_samples

    # Initialize training metrics
    metrics = metric(len(training_generator.dataset), device=device)

    # Initialize validation metrics if generator is provided.
    if val_generator is not None:
        metrics_val = metric(len(val_generator.dataset), device=device)

    # Initialize history arrays.
    history = []
    history_val = []

    # Initialize TQDM if verbose is set to 1.
    if verbose == 1:
        # Initialize TQDM bar
        tepoch = tqdm(range(epochs), unit="epoch")
        tepoch.set_description("Training ")
    else:
        tepoch = range(epochs)

    # Training loop
    for epoch in tepoch:
        # Set the model on training mode. Activate Dropout layers.
        model.train()
        # Set the number of training monte carlo samples
        model.num_samples = train_mc_samples
        # Mini-batches loop
        for data, target in training_generator:
            # Set batches in device
            data = data.to(device)
            target = target.to(device)
            # Compute loss value
            loss = model.train_step(optimizer, data, target)
            # Update metrics for the given batch. No gradients
            # are computed here in order to speed up the training.
            with torch.no_grad():
                # Make predictions
                mean_pred, std_pred = model(data)
                
                # Compute metrics using the original data scaled.
                scaled_target = target* model.y_std + model.y_mean
                #log_likelihood = model.predict_logdensity(data, scaled_target)
                metrics.update(
                    scaled_target,
                    loss,
                    mean_pred,
                    std_pred,
                    model.likelihood,
                )
        # Store history of metrics
        metrics_dict = metrics.get_dict()
        history.append(metrics_dict)
        # Reset current metrics for next epochs.
        metrics.reset()

        # Dictionary that stores validation metrics in tqdm.
        val_postfix = {}

        # Validation step
        if val_generator is not None:
            # Set the model in evaluation mode. Dropout layers
            # are off.
            model.eval()
            # Set the number of MC samples
            model.num_samples = val_mc_samples

            # Turn off gradients to speed-up the process
            with torch.no_grad():
                # Mini-batch loop
                for data, target in val_generator:
                    # Send to device
                    data = data.to(device)
                    target = target.to(device)
                    
                    # Get loss and predictions.
                    loss, mean_pred, std_pred = model.test_step(data, target)
                    #log_likelihood = model.predict_logdensity(data, target)
                    # Compute validation metrics
                    metrics_val.update(target, 
                                       loss,
                                       mean_pred, 
                                       std_pred,
                                       model.likelihood)

            # Store metrics and reset them
            metrics_val_dict = metrics_val.get_dict()
            metrics_val.reset()
            history_val.append(metrics_val_dict)

            # Handle Validation metrics in TQDM
            if verbose == 1:
                val_postfix = {k.lower()+'_val': v for k, v in metrics_val_dict.items()}

        # Show metrics in TQDM
        if verbose == 1:
            tepoch.set_postfix(
                {
                    **{k.lower()+'_train': v for k, v in metrics_dict.items()},
                    **val_postfix,
                }
            )

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step()

    # Return historial of metrics
    if val_generator is None:
        return history
    return history, history_val


def predict_prior_samples(model, generator, device=None):
    """
    Creates predictions of the prior for the given data generator.

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
    prior : numpy darray of shape (L, S, N, D)
            Contains the prior samples of the L layers, over the S
            mixture components on N points with output dimension D.
    """

    # No grands are computed in order to speed-up the process.
    with torch.no_grad():
        # Initialize the returning array
        prior = []
        # Mini-batch loop
        for data in generator:
            # Consider generators with no target values.
            try:
                batch_x, _ = data
            except:
                batch_x = data
            # Get prior samples from the model
            prior_samples = model.get_prior_samples(batch_x.to(device))

            # Append the results
            prior.append(prior_samples.detach().numpy())

        # Transform to numpy array concatenating on minibatch
        # dimension.
        prior = np.concatenate(prior, axis=2)

    return prior
