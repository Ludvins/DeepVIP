import numpy as np
import torch
import math
import torch.nn.functional as F


class Linear(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, device, dtype):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(
                torch.empty([output_dim, input_dim], dtype=dtype, device=device)
            )
        self.bias = torch.nn.Parameter(
                torch.empty([output_dim], dtype=dtype, device=device)
            )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def get_weights(self):
        return torch.cat(
            [self.weight.flatten(),
             self.bias.flatten()],
            dim = -1
            )
    

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
    
    def forward_weights(self, inputs, weights):
        shape = weights.shape[:-1]
        aux = self.input_dim * self.output_dim
        w = weights[..., :aux].reshape(*shape, *self.weight_mu.shape)
        b = weights[..., aux:aux + self.output_dim].reshape(*shape, *self.bias_mu.shape)

        return F.linear(input, w, b)

class MLP(torch.nn.Module):
    def __init__(
        self,
        structure,
        activation,
        input_dim,
        output_dim,
        device=None,
        dtype=torch.float64,
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim

        # Store parameters
        self.structure = structure
        self.activation = activation
        self.generator = torch.Generator()
        # Create an array symbolizing the dimensionality of the data at
        # each inner layer.
        dims = [self.input_dim] + structure + [output_dim]
        layers = []

        # Loop over the input and output dimension of each sub-layer.
        for _in, _out in zip(dims, dims[1:]):

            # Append the Bayesian linear layer to the array of layers
            layers.append(
                Linear(
                    _in,
                    _out,
                    device=device,
                    dtype=dtype,
                )
            )
        # Store the layers as ModuleList so that pytorch can handle
        # training/evaluation modes and parameters.
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Last layer has identity activation function
        return self.layers[-1](x)

    def forward_weights(self, inputs, weights):
        # raise NotImplementedError("Need to test forward weights")
        x = inputs

        pre = 0

        for i, _ in enumerate(self.layers[:-1]):

            post = pre + (self.layers[i].input_dim + 1) * self.layers[i].output_dim
            # Apply BNN layer
            x = self.activation(
                self.layers[i].forward_weights(x, weights[..., pre:post])
            )
            pre = post

        post = pre + (self.layers[-1].input_dim + 1) * self.layers[-1].output_dim
        # Last layer has identity activation function
        return self.layers[-1].forward_weights(x, weights[..., pre:post])

    def get_weights(self):
        weights = torch.cat(
            [layer.get_weights() for layer in self.layers],
            dim = -1
            )
        return weights



