#!/usr/bin/env python3

import torch

from torch.func import jacrev, jacfwd, vmap, functional_call
import copy

class TorchJacobian():
    def __init__(self, model, *args):
        fmodel, params_values = self.make_functional(model)
        self.raw_model = model
        self.model = fmodel
        self.params = params_values
        self.jac = jacrev(self.model)

    def make_functional(self, mod, disable_autograd_tracking=False):
        params_dict = dict(mod.named_parameters())
        params_names = params_dict.keys()
        params_values = tuple(params_dict.values())

        stateless_mod = copy.deepcopy(mod)
        stateless_mod.to('meta')

        def fmodel(new_params_values, *args, **kwargs):
            new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
            return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

        if disable_autograd_tracking:
            params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
        return fmodel, params_values


    def jacobians(self, x):
        jacobians = self.jac( self.params, x)
        jacobians = torch.cat([J.flatten(2, -1) for J in jacobians], -1)
        return jacobians, self.raw_model(x)



    def jacobians_on_outputs(self, x, outputs):
        # jacrev computes jacobians of argnums=0 by default.
        # We set it to 1 to compute jacobians of params
        jacobians = self.jac( self.params, x)
        jacobians = torch.cat([J.flatten(2, -1) for J in jacobians], -1)
        jacobians = torch.gather(jacobians, 1, outputs.unsqueeze(-1))
        f =  self.raw_model(x)

        return jacobians, f
