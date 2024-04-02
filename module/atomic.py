import math
import torch

from module.abstract import *


class Identity(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x


class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x.flatten(start_dim=1)


class Duplicate(Module):
    def __init__(self, num_copies):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x.unsqueeze(dim=-1).expand(list(x.shape)+[num_copies])


class Enumerate(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.arange(0, x.size()[1], dtype=torch.long, device=x.device)


class Abs(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.abs(x)


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = torch.nn.functional.relu


def ScaledReLU():
    return math.sqrt(2) * ReLU()


class MeanSubtract(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x - x.mean(dim=dim, keepdim=True)


class RMSDivide(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x / x.square().mean(dim=dim, keepdim=True).sqrt()


def LayerNorm(dim=-1):
    return RMSDivide(dim) @ MeanSubtract(dim)


class Mean(Module):
    def __init__(self, dim):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x.mean(dim=dim)


class AvgPool(Module):
    def __init__(self, output_size = (1,1)):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.nn.functional.adaptive_avg_pool2d(x, output_size)


def spectral_norm(p, u, num_steps=1):
    for _ in range(num_steps):
        u /= u.norm(dim=0, keepdim=True)
        v = torch.einsum('ab..., b... -> a...', p, u)
        u = torch.einsum('a..., ab... -> b...', v, p)
    return u.norm(dim=0, keepdim=True).sqrt(), u


class Linear(Module):
    def __init__(self, out_features, in_features, num_heads=None, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x):
        if self.num_heads is None: x = x.unsqueeze(dim=-1)
        x = self.scale * torch.einsum('ijh, ...jh -> ...ih', self.weight, x)
        if self.num_heads is None: x = x.squeeze(dim=-1)

        return x

    def initialize(self, device):
        num_heads = 1 if self.num_heads is None else self.num_heads
        self.weight = torch.empty((self.out_features, self.in_features, num_heads), device=device, requires_grad=True)
        for head in range(num_heads):
            torch.nn.init.orthogonal_(self.weight[:,:,head])
        self.parameters = [self.weight]

        self.momentum = torch.zeros_like(self.weight)
        self.u = torch.randn_like(self.weight[0])

    @torch.no_grad()
    def update(self, lr, hps):
        self.momentum += (1-hps["beta"]) * (self.weight.grad - self.momentum)
        norm, self.u.data = spectral_norm(self.momentum, self.u)

        if (norm == 0.0).any():
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * hps["wd"]

        self.weight.grad = None

    def print_submodules(self):
        print(f"Linear module of shape {(self.out_features, self.in_features)} with {self.num_heads} heads and mass {self.mass}.")


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
    def update(self, lr, hps):
        self.momentum += (1-hps["beta"]) * (self.weight.grad - self.momentum)
        norm, self.u.data = spectral_norm(self.momentum, self.u)

        if (norm == 0.0).any():
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * hps["wd"]

        self.weight.grad = None

    def print_submodules(self):
        print(f"Conv2D module of shape {(self.out_features, self.in_features, self.k, self.k)} and mass {self.mass}.")


class Embedding(Module):

    def __init__(self, num_embedding, embedding_dim, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.scale = math.sqrt(embedding_dim)

    def forward(self, x):
        return self.scale * torch.nn.functional.embedding(x, self.weight)

    def initialize(self, device):
        self.weight = torch.empty((self.num_embedding, self.embedding_dim), device=device, requires_grad=True)
        torch.nn.init.normal_(self.weight)
        self.weight.data /= self.weight.norm(dim=1, keepdim=True)
        self.parameters = [self.weight]

    @torch.no_grad()
    def update(self, lr, hps):
        norm = self.weight.grad.norm(dim=1, keepdim=True)
        self.weight -= lr * torch.nan_to_num(self.weight.grad / norm)
        self.weight /= self.weight.norm(dim=1, keepdim=True)
        self.weight.grad = None

    def print_submodules(self):
        print(f"Embedding module: {self.num_embedding} embeddings of size {self.embedding_dim}. Mass {self.mass}.")


class FunctionalAttention(Module):

    def __init__(self, context, causal):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1

        self.context = context
        self.causal = causal

    def forward(self, x):
        q, k, v = x

        att = torch.einsum('bcqh, bCqh -> bcCh', q, k) / q.size()[-2]
        if self.causal:
            T = q.size()[-3]
            att = att.masked_fill(self.mask[:,:T,:T,:] == 0, float('-inf'))

        p = torch.nn.functional.softmax(att, dim=-2)
        return torch.einsum('bcCh, bCvh -> bcvh', p, v)

    def initialize(self, device):
        if self.causal:
            T = self.context
            self.mask = torch.tril(torch.ones(T, T, device=device)).view(1, T, T, 1)
