import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import (
    DiagGGNExact,
    DiagGGNMC,
    KFAC,
    KFLR,
    SumGradSquared,
    BatchGrad,
)
from backpack.context import CTX

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.utils import Kron


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend."""

    def __init__(self, model, output_dim, last_layer=False, subnetwork_indices=None):
        super().__init__(model, "regression", last_layer, subnetwork_indices)
        extend(self._model)
        self.output_size = output_dim
        self.train_mode = True

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        with torch.set_grad_enabled(True):
            print(x.shape)
            model = extend(self.model, use_converter=True)
            model.eval()
            to_stack = []
            model.zero_grad()
            out = model(x)

            for i in range(self.output_size):

                with backpack(BatchGrad(), retain_graph=True):
                    if self.output_size > 1:
                        out[:, i].sum().backward(create_graph=True)
                    else:
                        out.sum().backward(create_graph=True)
                    to_cat = []
                    for param in model.parameters():
                        to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                        delattr(param, "grad_batch")
                    Jk = torch.cat(to_cat, dim=1)
                    if self.train_mode == False:
                        Jk = Jk.detach()
                to_stack.append(Jk)

        model.zero_grad()
        x.grad = None
        CTX.remove_hooks()
        _cleanup(model)
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2)
        else:
            return Jk.unsqueeze(-1).transpose(1, 2)

    def jacobians_on_outputs(self, x, outputs):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per given output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        outputs : torch.Tensor
            input data `(batch)` on compatible device with model.


        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters)`
        f : torch.Tensor
            output function `(batch)`
        """
        with torch.set_grad_enabled(True):
            model.eval()
            model = extend(self.model,  use_converter=True)
            to_stack = []
            model.zero_grad()

            out = model(x)
            for i in range(outputs.shape[1]):
                c = outputs[:, i]

                with backpack(BatchGrad(), retain_graph=self.train_mode):
                    torch.gather(out, 1, c.unsqueeze(-1)).sum().backward(create_graph=self.train_mode)
                    to_cat = []
                    for param in model.parameters():
                        to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                        delattr(param, "grad_batch")
                    Jk = torch.cat(to_cat, dim=1)
                    if self.subnetwork_indices is not None:
                        Jk = Jk[:, self.subnetwork_indices]
                    if self.train_mode == False:
                        Jk = Jk.detach()
                to_stack.append(Jk)
                

        model.zero_grad()
        x.grad = None
        CTX.remove_hooks()
        _cleanup(model)
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2)
        else:
            return Jk.unsqueeze(-1).transpose(1, 2)


def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
