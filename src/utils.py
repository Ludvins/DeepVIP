import torch

default_jitter = 1e-7


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
        raise NotImplementedError
