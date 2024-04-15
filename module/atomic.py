import math
import torch

from module.vector import Vector
from module.abstract import Module

def spectral_norm(p, num_steps=1):
    u = torch.randn_like(p[0])

    for _ in range(num_steps):
        v = torch.einsum('ab..., b... -> a...', p, u)
        v /= v.norm(dim=0, keepdim=True)
        u = torch.einsum('a..., ab... -> b...', v, p)

    return u.norm(dim=0, keepdim=True)


class Linear(Module):
    def __init__(self, out_features, in_features, num_heads=None, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x, w):
        if self.num_heads is None: x = x.unsqueeze(dim=-1)
        x = self.scale * torch.einsum('ijh, ...jh -> ...ih', w.weight, x)
        if self.num_heads is None: x = x.squeeze(dim=-1)

        return x

    def initialize(self, device):
        num_heads = 1 if self.num_heads is None else self.num_heads
        weight = torch.empty((self.out_features, self.in_features, num_heads), device=device, requires_grad=True)
        for head in range(num_heads):
            torch.nn.init.orthogonal_(weight[:,:,head])
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, vector):
        return Vector(vector.weight / spectral_norm(vector.weight))

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

    def forward(self, x, w):
        return self.scale * torch.nn.functional.conv2d(x, w.weight, None, self.stride, self.pad)

    def initialize(self, device):
        weight = torch.empty((self.out_channels, self.in_channels, self.k, self.k), device=device, requires_grad=True)
        for kx in range(self.k):
            for ky in range(self.k):
                torch.nn.init.orthogonal_(weight[:,:,kx,ky])
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, vector):
        return Vector(vector.weight / vector.weight.norm(dim=(0,1), keepdim=True))

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

    def forward(self, x, w):
        return self.scale * torch.nn.functional.embedding(x, w.weight)

    def initialize(self, device):
        weight = torch.empty((self.num_embedding, self.embedding_dim), device=device, requires_grad=True)
        torch.nn.init.normal_(weight)
        weight.data /= self.weight.norm(dim=1, keepdim=True)
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, vector):
        return Vector(vector.weight / vector.weight.norm(dim=1, keepdim=True))

    def print_submodules(self):
        print(f"Embedding module: {self.num_embedding} embeddings of size {self.embedding_dim}. Mass {self.mass}.")
