# Source gpflow
from typing import Iterable
import torch
import numpy as np
import itertools

def hermgauss(n, dtype):
    # Return the locations and weights of GH quadrature
    x, w = np.polynomial.hermite.hermgauss(n)
    return torch.tensor(x, dtype=dtype), torch.tensor(w, dtype=dtype)

def mvhermgauss(H, D, dtype):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    Parameters
    ----------
    H : integer
        Number of Gauss-Hermite evaluation points.
    D : integer
        Number of input dimensions. Needs to be known at call-time.
    """
    gh_x, gh_w =  np.polynomial.hermite.hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return torch.tensor(x, dtype= dtype), torch.tensor(w, dtype = dtype)


def ndiagquad(funcs, H, Fmu, Fvar, dtype, logspace=False, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Gauss-Hermite quadrature. The Gaussians must be independent.

    Parameters
    ----------
    funcs : Callable or Iterable of Callables
            The function integrands. Operates elementwise. 
            All arguments will be tensors of shape (N, 1).
    H : int
        number of Gauss-Hermite quadrature points
    Fmu : Tenfor of shape

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise, on the following arguments:
        - `Din` positional arguments to match Fmu and Fvar; i.e., 1 if Fmu and Fvar are tensors;
          otherwise len(Fmu) (== len(Fvar)) positional arguments F1, F2, ...
        - the same keyword arguments as given by **Ys
        All arguments will be tensors of shape (N, 1)

    :param H: number of Gauss-Hermite quadrature points
    :param Fmu: array/tensor or `Din`-tuple/list thereof
    :param Fvar: array/tensor or `Din`-tuple/list thereof
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N` (i.e., shape (N,) or (N, 1))
    :return: shape is the same as that of the first Fmu
    """
    def unify(f_list):
        """
        Stack a list of means/vars into a full block
        """
        return torch.reshape(
                torch.concat([torch.reshape(f, (-1, 1)) for f in f_list], axis=1),
                (-1, 1, Din))

    if isinstance(Fmu, (tuple, list)):
        Din = len(Fmu)
        shape = torch.shape(Fmu[0])
        Fmu, Fvar = map(unify, [Fmu, Fvar])    # both N x 1 x Din
    else:
        Din = 1
        shape = Fmu.shape
        Fmu, Fvar = [torch.reshape(f, (-1, 1, 1)) for f in [Fmu, Fvar]]

    xn, wn = mvhermgauss(H, Din, dtype)
    # xn: H**Din x Din, wn: H**Din

    gh_x = xn.reshape(1, -1, Din)             # 1 x H**Din x Din
    Xall = gh_x * torch.sqrt(2.0 * Fvar) + Fmu   # N x H**Din x Din
    Xs = [Xall[:, :, i] for i in range(Din)]  # N x H**Din  each

    gh_w = wn * np.pi ** (-0.5 * Din)  # H**Din x 1

    for name, Y in Ys.items():
        Y = torch.reshape(Y, (-1, 1))
        Y = torch.tile(Y, [1, H**Din])  # broadcast Y to match X
        # without the tiling, some calls such as tf.where() (in bernoulli) fail
        Ys[name] = Y  # now N x H**Din

    def eval_func(f):
        feval = f(*Xs, **Ys)  # f should be elementwise: return shape N x H**Din
        if logspace:
            log_gh_w = np.log(gh_w.reshape(1, -1))
            result = torch.reduce_logsumexp(feval + log_gh_w, axis=1)
        else:
            result = torch.matmul(feval, gh_w.reshape(-1, 1))
        return torch.reshape(result, shape)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)