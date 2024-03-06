import math
import torch

from module.abstract import Module


class Identity(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x
        self.initialize = lambda : None


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = torch.nn.functional.relu
        self.initialize = lambda : None


class Linear(Module):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)
    
    def norm(self, w):
        return torch.linalg.norm(w, ord=2) / self.scale

    def initialize(self):
        self.weight = torch.nn.Parameter(torch.empty((self.out_features, self.in_features))) 
        torch.nn.init.orthogonal_(self.weight)
        self.weight.data *= self.scale

        self.register_buffer("momentum", torch.zeros_like(self.weight))
        self.register_buffer("u",        torch.randn_like(self.weight[0]))

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.momentum += (1-beta) * (self.weight.grad - self.momentum)
        self.u = self.momentum @ self.u @ self.momentum / self.u.norm()
        self.weight -= lr * self.momentum / self.u.norm().sqrt() * self.scale
        self.weight *= 1 - lr * wd
