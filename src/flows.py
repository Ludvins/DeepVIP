#!/usr/bin/env python3

import torch
from .generative_functions import BayesianNN, BayesLinear, SimplerBayesLinear


class Flow(torch.nn.Module):
    def __init__(self, device, dtype, seed):
        self.device = device
        self.dtype = dtype
        self.seed = seed
        super().__init__()

    def forward(x):
        raise NotImplementedError


class SimpleFlow(Flow):
    def __init__(self, depth, device, dtype, seed):
        self.depth = depth
        super().__init__(device, dtype, seed)

        params = torch.cat(
            (
                torch.ones(depth, 1),
                torch.zeros(depth, 1),
                torch.ones(depth, 1),
                torch.zeros(depth, 1),
            ),
            axis=-1,
        )
        self.params = torch.nn.Parameter(params)

    def forward(self, f, x=None):
        for param in self.params:
            f = param[0] * torch.sinh(param[2] * torch.arcsinh(f) - param[3]) + param[1]

        return f


class InputDependentFlow(Flow):
    def __init__(self, depth, input_dim, device, dtype, seed):
        self.depth = depth
        self.input_dim = input_dim
        super().__init__(device, dtype, seed)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 50, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 4 * self.depth, dtype=dtype),
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        # self.nn[4].weight.data.fill_(0)
        # self.nn[4].bias.data.fill_(0)
        # self.nn.apply(init_weights)

    def forward(self, f, x):
        need_squeeze = False
        if x.shape[-1] != self.input_dim and self.input_dim == 1:
            x = x.unsqueeze(-1)
            need_squeeze = True

        params = self.nn(x)
        params = params.reshape((self.depth, 4, *x.shape))
        for param in params:
            f = (1 - param[0]) * torch.sinh(
                (1 - param[2]) * torch.arcsinh(f) - param[3]
            ) + param[1]
            # f = f * torch.exp(param[0]) + param[1]

        if need_squeeze:
            f = f.squeeze(-1)
        return f


class InputDependentFlow2(Flow):
    def __init__(self, depth, input_dim, device, dtype, seed):
        self.depth = depth
        self.input_dim = input_dim
        super().__init__(device, dtype, seed)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 10, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 4 * self.depth, dtype=dtype),
        )

        # self.nn = BayesianNN(
        #     [50],
        #     torch.tanh,
        #     1,
        #     input_dim,
        #     4 * self.depth,
        #     BayesLinear,
        #     fix_random_noise=False,
        #     device=device,
        #     dtype=dtype,
        # )
        # for layer in self.nn.layers:
        #     layer.weight_log_sigma = torch.nn.Parameter(layer.weight_log_sigma - 5)
        #     layer.bias_log_sigma = torch.nn.Parameter(layer.bias_log_sigma - 5)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        self.nn[-1].weight.data.fill_(0)
        self.nn[-1].bias.data.fill_(0)
        # self.nn.apply(init_weights)

    def forward(self, a):
        params = self.nn(a)
        params = params.reshape((self.depth, 4, *a.shape))
        f = a
        for param in params:
            f = (1 - param[0]) * torch.sinh(
                (1 - param[2]) * torch.arcsinh(f) - param[3]
            ) + param[1]
            # f = f * torch.exp(param[0]) + param[1]

        return f

    def KL(self):
        l2 = 0
        l2 += torch.mean(self.nn[0].weight ** 2)
        l2 += torch.mean(self.nn[0].bias ** 2)
        l2 += torch.mean(self.nn[2].weight ** 2)
        l2 += torch.mean(self.nn[2].bias ** 2)
        return l2
