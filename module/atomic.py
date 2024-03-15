import math
import torch

from module.abstract import Module


class Identity(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x
        self.initialize = lambda device: None


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = torch.nn.functional.relu
        self.initialize = lambda device: None


class MeanSubtract(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x - x.mean(dim=1, keepdim=True)
        self.initialize = lambda device: None


class Abs(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.abs(x)
        self.initialize = lambda device: None


def ScaledReLU():
    return math.sqrt(2) * ReLU()


class Linear(Module):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x):
        return self.scale * torch.nn.functional.linear(x, self.weight)

    def initialize(self, device):
        self.weight = torch.empty((self.out_features, self.in_features), device=device, requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)
        self.parameters = [self.weight]

        self.momentum = torch.zeros_like(self.weight)
        self.u = torch.randn_like(self.weight[0])

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.momentum += (1-beta) * (self.weight.grad - self.momentum)
        self.u = self.momentum @ self.u @ self.momentum / self.u.norm()

        if (norm := self.u.norm().sqrt()) == 0.0:
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * wd

        self.weight.grad = None
