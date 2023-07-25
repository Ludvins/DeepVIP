
import torch

class NTK():
    def __init__(self, model) -> None:
        self.model = model
        
    def cached_forward(self, x):
        cached_values = []
        for layer in self.model.layers():
            cached_values.append(x)
            x = layer(x)
            
        return x, cached_values
    
    def __forward__(self, x):
        return self.model(x)
    
    
    def linear_backward(dout, cache, layer):
        return dout @ layer.weights.T
    
    def tanh_backward(dout, cache, layer):
        x = cache
        derivative = 1 - torch.tanh(x)**2

        return torch.einsum("a...p, ap -> a...p", dout, derivative)