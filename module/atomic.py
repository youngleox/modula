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


class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x.flatten(start_dim=1)
        self.initialize = lambda device: None


class Abs(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.abs(x)
        self.initialize = lambda device: None


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = torch.nn.functional.relu
        self.initialize = lambda device: None


def ScaledReLU():
    return math.sqrt(2) * ReLU()


class MeanSubtract(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x - x.mean(dim=tuple(range(1,x.dim())), keepdim=True)
        self.initialize = lambda device: None


class RMSDivide(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x / x.square().mean(dim=tuple(range(1,x.dim())), keepdim=True).sqrt()
        self.initialize = lambda device: None


def LayerNorm():
    return RMSDivide() @ MeanSubtract()


class AvgPool(Module):
    def __init__(self, output_size = (1,1)):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.nn.functional.adaptive_avg_pool2d(x, output_size)
        self.initialize = lambda device: None


def spectral_norm(p, u, num_steps=1):
    for _ in range(num_steps):
        u /= u.norm(dim=0, keepdim=True)
        v = torch.einsum('ab..., b... -> a...', p, u)
        u = torch.einsum('a..., ab... -> b...', v, p)
    return u.norm(dim=0, keepdim=True).sqrt(), u


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
        norm, self.u.data = spectral_norm(self.momentum, self.u)

        if norm == 0.0:
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * wd

        self.weight.grad = None


class Conv2D(Module):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1, padding=1, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.k = kernel_size
        self.stride = stride
        self.pad = padding
        self.scale = math.sqrt(out_channels / in_channels) / (kernel_size ** 2)

    def forward(self, x):
        return self.scale * torch.nn.functional.conv2d(x, self.weight, None, self.stride, self.pad)

    def initialize(self, device):
        self.weight = torch.empty((self.out_channels, self.in_channels, self.k, self.k), device=device, requires_grad=True)
        for kx in range(self.k):
            for ky in range(self.k):
                torch.nn.init.orthogonal_(self.weight[:,:,kx,ky])
        self.parameters = [self.weight]

        self.momentum = torch.zeros_like(self.weight)
        self.u = torch.randn_like(self.weight[0])

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.momentum += (1-beta) * (self.weight.grad - self.momentum)
        norm, self.u.data = spectral_norm(self.momentum, self.u)

        if (norm == 0.0).any():
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * wd

        self.weight.grad = None
