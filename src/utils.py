import torch

default_jitter = 1e-7

import numpy as np


def smooth(x, window_len=11):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 2:
        return x
    
    n = x.shape[0]
    n_bins = n//window_len
    
    if n % window_len != 0:
        n_bins += 1
    
    y = np.zeros(n_bins)
    
    for i in range(n):
        y[i // window_len] += x[i]
        
    
    y = y / window_len
    
    if n % window_len != 0:
        y[-1] = y[-1] * window_len / (n % window_len)
    
    
    return y


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the `re-parameterization trick` for the Gaussian distribution.
    The covariance matrix can be either complete or diagonal.

    Parameters
    ----------
    mean : tf.tensor of shape (N, D)
           Contains the mean values for each Gaussian sample
    var : tf.tensor of shape (N, D) or (N, N, D)
          Contains the covariance matrix (either full or diagonal) for
          the Gaussian samples.
    z : tf.tensor of shape (N, D)
        Contains a sample from a Gaussian distribution, ideally from a
        standardized Gaussian.
    full_cov : boolean
               Wether to use the full covariance matrix or diagonal.
               If true, var must be of shape (N, N, D) and full covariance
               is used. Otherwise, var must be of shape (N, D) and the
               operation is done elementwise.

    Returns
    -------
    sample : tf.tensor of shape (N, D)
             Sample of a Gaussian distribution. If the samples in z come from
             a Gaussian N(0, I) then, this output is a sample from N(mean, var)
    """
    # If no covariance values are given, the mean values are used.
    if var is None:
        return mean

    # Diagonal covariances -> Pointwise scale
    if full_cov is False:
        return mean + z * torch.sqrt(var + default_jitter)
    # Full covariance matrix
    else:
        var = torch.transpose(var, 0, 2)
        # Shape (..., N, N)
        L = torch.linalg.cholesky(var + default_jitter * torch.eye(var.shape[-1]))
        ret = torch.einsum("...nm,am...->an...", L, z)
        return mean + ret


def _del_nested_attr(obj, names) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj, names, value) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod):
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = []
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        if "mu" in name:
            _del_nested_attr(mod, name.split("."))
            names.append(name)
            orig_params.append(p)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod, names, params) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)
