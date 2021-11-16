import tensorflow as tf

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
        return mean + z * tf.math.sqrt(var + default_jitter)
    # Full covariance matrix
    else:
        # Get shapes
        N = tf.shape(mean)[0]
        D = tf.shape(mean)[1]

        # Set input dimension as second in shape order
        # Mean shape from SND to SDN
        mean = tf.transpose(mean, (1, 0))
        # Var shape from SNND to SDNN
        var = tf.transpose(var, (2, 0, 1))

        # Define a `default_jitter` matrix. Shape (1, N, N)
        I = default_jitter * tf.eye(N, dtype=tf.float64)[None, :, :]
        # Compute the Cholesky decomposition of the fluctuated variance matrix.
        # Shape (D, N, N)
        chol = tf.linalg.cholesky(var + I)

        # Change the shape of z from (N, D) to (D, N, 1)
        reshaped_z = tf.transpose(z, [1, 0])[:, :, None]
        # Compute the new samples using the Cholesky decomposition.
        # Shape (D, N). In fact the matmul operation returns (D, N, 1),
        # that's why the last dimension is instantiated.
        f = mean + tf.matmul(chol, reshaped_z)[:, :, 0]
        # Change to original shape order, from (D, N) to (N, D)

        return tf.transpose(f, (1, 0))
