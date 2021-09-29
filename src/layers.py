import tensorflow as tf

from utils import reparameterize

from tensorflow.python.keras.backend import batch_normalization
import tensorflow_probability as tfp
import numpy as np



class Layer(tf.keras.layers.Layer):
    def __init__(self, input_prop_dim=None, dtype=None, **kwargs):
        """
        A base class for VIP layers. Basic functionality for multisample 
        conditional and input propagation.
        
        Parameters
        ----------
        input_prop_dim : int
                         The first dimensions of X to propagate. 
                         If None (or zero) then no input propapagation is done
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        **kwargs : dict, optional
                   Extra arguments to `Layer`.
                   Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype=dtype, **kwargs)
        self.input_prop = input_prop_dim
        self.build(input_prop_dim)

    def conditional_ND(self, X, full_cov=False):
        """
        TODO
        
        Parameters
        ----------
        X : 
                 
        full_cov : 
        
        """
        raise NotImplementedError

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        """
        raise NotImplementedError

    @tf.function
    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional.
        
        Parameters
        ----------
        X : tf.tensor of shape (S, N, D_in)
            Input locations. 
            S independent draws of N samples with dimension D_in
        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.
        
        Returns
        -------
        mean : tf.tensor of shape (S, N, self.num_outputs)
               Stacked tensor of mean values from conditional_ND applied to 
               each sample.
        var : tf.tensor of shape (S, N, self.num_outputs or 
                                  S, N, N, self.num_outputs)
              Stacked tensor of variance values from conditional_ND applied to 
              each sample.
        """
        if full_cov:
            f = lambda a: self.conditional_ND(a, full_cov = True)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S = tf.shape(X)[0]
            N = tf.shape(X)[1]
            D = tf.shape(X)[2] 

            X_flat = tf.reshape(X, [S*N, D])
            mean, var = self.conditional_ND(X_flat, full_cov = False)
            
            # TODO Check this return shape
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    @tf.function
    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample.
        Adds input propagation if necessary.
        
        Parameters
        ----------
        X : tf.tensor of shape (S, N, D_in)
            Input locations. 
            S independent draws of N samples with dimension D_in

        z : tf.tensor of shape (S, N, D)
            Contains a sample from a Gaussian distribution, ideally from a
            standardized Gaussian.
        
        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.
        
        
        Returns
        -------
        samples : tf.tensor of shape (S, N, self.num_outputs)
                  Samples from a Gaussian given by mean and var. See below.
        mean : tf.tensor of shape (S, N, self.num_outputs)
               Stacked tensor of mean values from conditional_ND applied to X.
        var : tf.tensor of shape (S, N, self.num_outputs or 
                                  S, N, N, self.num_outputs)
              Stacked tensor of variance values from conditional_ND applied to 
              X.
        """
        mean, var = self.conditional_SND(X, full_cov = full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = tf.shape(X)[2]

        # TODO Check mean and var shapes
        
        # If no sample is given, generate it from a standardized Gaussian
        if z is None:
            z = tf.random.normal(shape=tf.shape(mean), seed=self.seed, 
                                 dtype=self.dtype)
        # Apply re-parameterization trick to z
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        # Apply input propagation if necessary
        if self.input_prop_dim:
            # Get the slice of X to propagate
            shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            # Add it to the samples and mean values
            samples = tf.concat([X_prop, samples], 2)
            mean = tf.concat([X_prop, mean], 2)

            # Add the corresponding zeros to the variance array
            if full_cov:
                shape = (tf.shape(X)[0], tf.shape(X)[1], 
                         tf.shape(X)[1], tf.shape(var)[3])
                zeros = tf.zeros(shape, dtype=self.dtype)
                var = tf.concat([zeros, var], 3)
            else:
                var = tf.concat([tf.zeros_like(X_prop), var], 2)

        return samples, mean, var


class VIPLayer(Layer):
    def __init__(self, noise_sampler, generative_f, regression_coeffs, 
                 num_outputs, mean_function, dtype = tf.float64, **kwargs):
        """
        A variational implicit process layer in whitened representation. 
        This layer holds the kernel, variational parameters, 
        linear regression coefficients and mean function.

        The underlying model performs a Bayesian linear regression
        approximation
        \[
            f(x) = mean_function(x) + a^T \phi(x)
        \]
        with
        \[
            \phi(x) = \frac{1}{\sqrt(S)}(f_1(x) - m(x), \dots, f_S(x) - m(x))   
        \]
        \[
            a \sim \mathcal{N}(0, I)
        \]
        Where S randomly samples functions are used and m(x) denotes 
        their mean.
        
        The variational distribution over the regression coefficients is
        \[
            Q(a) = \mathcal{N}(a_mu, a_sqrt a_sqrt^T)
        \]

        The layer holds D_out independent VIPs with the same kernel
        and regression coefficients.
        
        
        Parameters
        ----------
        kernel : The kernel function for the VIP layer. The input dimension 
                 is D_in.
        
        a : tf.tensor of shape (num_coeffs, D_in)
            Linear regression coefficients.
        
        num_outputs : int 
                      The number of independent VIP in this layer.
                      More precisely, q_mu has shape (S, num_outputs)
        
        mean_function : callable
                        Mean function added to the model.
        
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        **kwargs : dict, optional
                   Extra arguments to `Layer`.
                   Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype = dtype, **kwargs)
        self.num_coeffs = regression_coeffs.shape[0]

        # Regression Coefficients prior mean
        a_mu = np.zeros((self.num_coeffs, num_outputs))
        self.q_mu = tf.Variable(a_mu, name="a_mu")

        # Define tf variable with coefficients
        self.regression_coeffs = tf.Variable(regression_coeffs, 
                                             name = "regression_coeffs", 
                                             dtype = self.dtype)

        # Store class members
        self.mean_function = mean_function
        self.num_outputs = tf.constant(num_outputs, dtype=tf.int32)
        self.noise_sampler = noise_sampler
        self.generative_f = generative_f

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        I = np.eye(self.num_coeffs)
        # Replicate it num_outputs times
        # Shape (num_outputs, num_coeffs, num_coeffs)
        I_tiled = np.tile(I[None,:,:], [num_outputs,1,1])
        # Create tensor with triangular representation.
        # Shape (num_outputs, num_coeffs*(num_coeffs + 1)/2)
        triangular_I_tiled = tfp.math.fill_triangular_inverse(I_tiled)
        self.q_sqrt_tri = tf.Variable(triangular_I_tiled, name="q_sqrt_tri")
        

    @tf.function
    def conditional_ND(self, X, full_cov=False):
        """
        Computes Q*(y|x, a) using the linear regression approximation.
        
        Parameters
        ----------
        X : 
        
        full_cov : 
        
        Returns
        -------

        """
        
        # Generate function samples
        Z = self.noise_sampler(self.num_coeffs)
        f = self.generative_f(X, Z)
        
        # Compute mean value
        m = tf.reduce_mean(f, axis = 0)
        
        # TODO Check shapes and dimensions in operations
        # Compute regresion function
        phi = 1/tf.math.sqrt(self.num_coeffs) * (f - m)

        # Mean and var values
        mean = m + tf.matmul(phi, self.regression_coeffs)
        var = 
        
        return mean + self.mean_function(X), var


    @tf.function
    def KL(self):
        """
        Computes the KL divergence from the variational distribution of
        the linear regression coefficients to the prior.
        
        That is from a Gaussian N(a_mu, a_sqrt) to N(0, I).
        Uses formula for computing KL divergence between two
        multivariate normals, which in this case is:
        \[
            KL = 0.5* ( tr(a_sqrt^T a_sqrt) + 
                 a_mu^T a_mu - M - \log |a_sqrt^T a_sqrt| )   
        \]
        """

        D = tf.cast(self.num_outputs, dtype=self.dtype)

        # Recover triangular matrix from array
        a_sqrt = tfp.math.fill_triangular(self.a_sqrt_tri)

        # Constant dimensionality term
        KL = -0.5 * D * self.num_inducing
        
        # Log of determinant of covariance matrix. 
        # Uses that sqrt(det(X)) = det(X^(1/2)) and
        # that the determinant of a upper triangular matrix (which a_sqrt is),
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        KL -= 0.5 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(a_sqrt) ** 2))

        # Trace term
        KL += 0.5 * tf.reduce_sum(tf.linalg.diag_part(tf.square(a_sqrt)))
        
        # Mean term
        KL += 0.5 * tf.reduce_sum(self.a_mu**2)

        return KL

