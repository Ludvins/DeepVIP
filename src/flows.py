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

        self.params = torch.nn.Parameter(
            torch.zeros(self.depth, 4, dtype=self.dtype, device=self.device)
        )

    def forward(self, a):
        f = a
        for param in self.params:
            f = (1 - param[0]) * torch.sinh(
                (1 - param[2]) * torch.arcsinh(f) - param[3]
            ) + param[1]
            # f = f * torch.exp(param[0]) + param[1]

        return f

    def KL(self):
        return torch.sum(self.params ** 2)


class InputDependentFlow2(Flow):
    def __init__(self, depth, input_dim, device, dtype, seed):
        self.depth = depth
        self.input_dim = input_dim
        super().__init__(device, dtype, seed)
        self.nn = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(input_dim, 10, dtype=dtype),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(10, 10, dtype=dtype),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(10, 4 * self.depth, dtype=dtype),
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        self.nn[-1].weight.data.fill_(0)
        self.nn[-1].bias.data.fill_(0)
        # self.nn.apply(init_weights)

    def forward(self, a):
        params = self.nn(a)
        params = params.reshape((a.shape[0], self.depth, 4, 1))
        params = torch.permute(params, (1, 2, 0, 3))
        f = a
        for param in params:
            f = (1 - param[0]) * torch.sinh(
                (1 - param[2]) * torch.arcsinh(f) - param[3]
            ) + param[1]
            # f = f * torch.exp(param[0]) + param[1]

        return f

    def KL(self):
        l2 = 0
        # l2 += torch.sum(self.nn[-3].weight ** 2)
        # l2 += torch.sum(self.nn[-3].bias ** 2)
        l2 += torch.sum(self.nn[-1].weight ** 2)
        l2 += torch.sum(self.nn[-1].bias ** 2)
        return l2


class SAL(InputDependentFlow2):
    def __init__(self, depth, input_dim, device, dtype, seed):
        super().__init__(depth, input_dim, device, dtype, seed)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim // 2, 50, dtype=dtype),
            # torch.nn.Tanh(),
            # torch.nn.Linear(100, 100, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(50, 4 * depth, dtype=dtype),
        )
        self.nn[-1].weight.data.fill_(0)
        self.nn[-1].bias.data.fill_(0)

    def forward(self, a):
        print(a.shape)
        params = self.nn(a)[0]
        params = params.reshape((a.shape[0], self.depth, 4, 1))
        params = torch.permute(params, (1, 2, 0, 3))
        f = a
        for param in params:
            f = (1 - param[0]) * torch.sinh(
                (1 - param[2]) * torch.arcsinh(f) - param[3]
            ) + param[1]
            # f = f * torch.exp(param[0]) + param[1]

        return f

    def KL(self):
        l2 = 0
        return l2


class CouplingLayer(torch.nn.Module):
    def __init__(self, input_dim, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.input_dim = input_dim

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim // 2, 100, dtype=dtype),
            torch.nn.Tanh(),
            # torch.nn.Linear(100, 100, dtype=dtype),
            # torch.nn.ReLU(),
            torch.nn.Linear(100, input_dim, dtype=dtype),
        )
        self.nn[-1].weight.data.fill_(0)
        self.nn[-1].bias.data.fill_(0)

    def forward(self, a):
        z1 = a[:, : self.input_dim // 2]
        z2 = a[:, self.input_dim // 2 :]
        nn = self.nn(z1)  # [0]
        mu = nn[:, : self.input_dim // 2]
        sigma = nn[:, self.input_dim // 2 :]
        z2 = z2 * torch.exp(sigma) + mu

        ldj = torch.sum(sigma, axis=1)
        return torch.cat([z1, z2], dim=1), ldj


class CouplingFlow(Flow):
    def __init__(self, depth, input_dim, device, dtype, seed):
        self.depth = depth
        self.input_dim = input_dim
        super().__init__(device, dtype, seed)

        self.generator = torch.Generator(device)
        self.generator.manual_seed(000)
        biyections = []

        for _ in range(depth):
            biyections.append(CouplingLayer(input_dim, device, dtype))

        self.biyections = torch.nn.ModuleList(biyections)

    def forward(self, a):
        LDJ = 0
        for b in self.biyections:
            a, ldj = b(a)
            a = a.flip(-1)
            LDJ += ldj
        return a, -LDJ
