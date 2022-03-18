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
    gh_x, gh_w = np.polynomial.hermite.hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return torch.tensor(x, dtype=dtype), torch.tensor(w, dtype=dtype)


def hermgaussquadrature(f, num_gh, Fmu, Fvar, Y, dtype):

    # Shape (num hermite)
    xn, wn = hermgauss(num_gh, dtype)

    # Shape (1, num_hermite)
    gh_x = xn.reshape(1, -1)
    # Shape (N, num_hermite)
    Xall = gh_x * torch.sqrt(torch.clip(2.0 * Fvar, min=1e-10)) + Fmu

    # Shape (num_hermite, 1)
    gh_w = wn.reshape(-1, 1) / np.sqrt(np.pi)

    if Y is not None:
        # Shape( N, num_hermite )
        Y = torch.tile(Y, [1, num_gh])  # broadcast Y to match X
        feval = f(Xall, Y)
    else:
        feval = f(Xall)

    # Shape (N, num_hermite)
    feval = feval @ gh_w
    return feval
