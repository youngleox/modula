import math
import torch

from module.abstract import Module
from module.vector import Vector


class Bond(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.length = 0
        self.initialize = lambda device : Vector()
        self.normalize  = lambda w, target_norm : None
        self.regularize = lambda w, strength : None


class Identity(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x


class Flatten(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.flatten(start_dim=1)


class Duplicate(Bond):
    def __init__(self, num_copies):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.unsqueeze(dim=-1).expand(list(x.shape)+[num_copies])


class Enumerate(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.arange(0, x.size()[1], dtype=torch.long, device=x.device)


class Abs(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.abs(x)


class ReLU(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.relu(x)


def ScaledReLU():
    return math.sqrt(2) * ReLU()


class GELU(Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.gelu(x)


def ScaledGELU():
    return math.sqrt(2) * GELU()


class MeanSubtract(Bond):
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x - x.mean(dim=dim, keepdim=True)


class RMSDivide(Bond):
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x / x.square().mean(dim=dim, keepdim=True).sqrt()


def LayerNorm(dim=-1):
    return RMSDivide(dim) @ MeanSubtract(dim)


class Mean(Bond):
    def __init__(self, dim):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.mean(dim=dim)


class AvgPool(Bond):
    def __init__(self, output_size = (1,1)):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.nn.functional.adaptive_avg_pool2d(x, output_size)


class FunctionalAttention(Bond):

    def __init__(self, causal):
        super().__init__()
        self.sensitivity = 1
        self.causal = causal

    def forward(self, x, w):
        q, k, v = x

        att = torch.einsum('bcqh, bCqh -> bcCh', q, k) / q.size()[-2]
        if self.causal:
            att -= torch.ones_like(att[0,:,:,0]).mul_(float('inf')).triu_(diagonal=1).unsqueeze(0).unsqueeze(-1)
        p = torch.nn.functional.softmax(att, dim=-2)
        y = torch.einsum('bcCh, bCvh -> bcvh', p, v)

        return y
