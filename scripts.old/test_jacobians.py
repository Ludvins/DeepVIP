import sys

sys.path.append(".")


from src.backpack_interface import BackPackInterface

from utils.models import get_mlp, MLP
import torch

input_dim = 10
output_dim = 10
X = torch.rand(size=(5, input_dim), dtype=torch.float64)
Z = torch.rand(size=(5, input_dim), dtype=torch.float64)

mlp_model = get_mlp(
    input_dim, output_dim, [50, 50], torch.nn.Tanh, device="cpu", dtype=torch.float64
)
backend = BackPackInterface(mlp_model, mlp_model.output_size)
ad_jacobians = backend.jacobians(X)
print(ad_jacobians.shape)
ad_jacobians_z = backend.jacobians(Z)


mlp_model_adhoc = MLP(input_dim, output_dim, 2, 50, device="cpu", dtype=torch.float64)


mlp_model_adhoc.layers[0].weight = torch.nn.Parameter(mlp_model[0].weight.T)
mlp_model_adhoc.layers[0].bias = mlp_model[0].bias
mlp_model_adhoc.layers[2].weight = torch.nn.Parameter(mlp_model[2].weight.T)
mlp_model_adhoc.layers[2].bias = mlp_model[2].bias
mlp_model_adhoc.layers[4].weight = torch.nn.Parameter(mlp_model[4].weight.T)
mlp_model_adhoc.layers[4].bias = mlp_model[4].bias
print("")
input("Type anything")

print("Testing both models have the same output: ", end="")
print(torch.equal(mlp_model(X), mlp_model_adhoc(X)))


ad_kernel = torch.einsum("nap, mbp -> nmab", ad_jacobians, ad_jacobians)

ad_kernel_xz = torch.einsum("nap, mbp -> nmab", ad_jacobians, ad_jacobians_z)

ad_kernel_zz = torch.einsum("nap, mbp -> nmab", ad_jacobians_z, ad_jacobians_z)

manual_kernel = mlp_model_adhoc.get_kernel(X, X)


def test_equal(a, b):
    diff = a - b
    sum = torch.sum(torch.abs(diff))
    return sum.item() < 1e-6


print("Testing both models have the same Jacobians at first output dimension: ", end="")
J = mlp_model_adhoc.get_jacobian_on_outputs(X, torch.zeros(X.shape[0]).to(torch.long))
print(test_equal(torch.sum(ad_jacobians[:, 0]), torch.sum(J)))

print("Testing both models have the same Jacobians: ", end="")
J = mlp_model_adhoc.get_jacobian(X)
print(test_equal(torch.sum(ad_jacobians), torch.sum(J)))

print("Testing both models have the same kernel at the first point: ", end="")
print(test_equal(ad_kernel[0][0], manual_kernel[0][0]))

print("Testing both models have the same kernel diagonal: ", end="")
print(
    test_equal(
        torch.diagonal(ad_kernel, dim1=0, dim2=1),
        torch.diagonal(manual_kernel, dim1=0, dim2=1),
    )
)

print("Testing both models have the same kernel: ", end="")
print(test_equal(ad_kernel, manual_kernel))

x, Kx, Kxz, Kzz = mlp_model_adhoc.get_full_kernels(
    X, X, torch.zeros(X.shape[0], dtype=torch.int64)
)

print("Testing both models have the same kernel diagonal: ", end="")
print(test_equal(torch.diagonal(ad_kernel, dim1=0, dim2=1).permute(2, 0, 1), Kx))

#x, Kx, Kxz, Kzz = mlp_model_adhoc.get_full_kernels2(X, Z)
Kxz = mlp_model_adhoc.get_kernel(X, Z)
print("Testing both models have the same kernel: ", end="")
print(test_equal(ad_kernel_xz, Kxz))

Kzz = mlp_model_adhoc.get_kernel(Z, Z)
print("Testing both models have the same kernel diagonal: ", end="")
print(test_equal(ad_kernel_zz, Kzz))
