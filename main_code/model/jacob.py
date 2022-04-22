import torch.nn as nn
import torch.autograd as autograd
from torch.autograd.functional import vjp 

class jacobinNet(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""

    def __init__(self, dnn: nn.Module):
        super(jacobinNet, self).__init__()
        self.dnn = dnn

    def forward(self, x, create_graph=True, strict=True):
        def f(x): return self.dnn(x).mean([1, 2, 3]).sum()
        J = autograd.functional.jacobian(
            f, x, create_graph=create_graph, strict=strict)
        return J

class VJPNet(nn.Module):
    def __init__(self, dnn: nn.Module):
        super(jacobinNet, self).__init__()
        self.dnn = dnn

    def forward(self, x, create_graph=True, strict=True):
        def f_(x): return self.dnn(x).mean([1, 2, 3]).sum()
        def f(x): return autograd.functional.jacobian(f_, x, create_graph=create_graph, strict=strict)
        vjVector=vjp(f,x,create_graph=create_graph, strict=strict)
        return vjVector