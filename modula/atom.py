import math
import torch

from modula.abstract import Module
from modula.vector import Vector


class Linear(Module):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1

        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x, w):
        return self.scale * torch.nn.functional.linear(x, w[0])

    def initialize(self, device):
        weight = torch.empty((self.out_features, self.in_features), device=device, requires_grad=True)
        torch.nn.init.orthogonal_(weight)
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, w, target_norm):
        weight = w[0]
        if not hasattr(self, "u"):
            self.u = torch.randn_like(weight[0])
            self.v = torch.empty_like(weight[:,0])
        torch.mv(weight, self.u, out=self.v)
        self.v /= self.v.norm()
        torch.mv(weight.t(), self.v, out=self.u)
        weight *= target_norm / self.u.norm()

    @torch.no_grad()
    def regularize(self, w, strength):
        weight = w[0]
        weight *= 1 - strength

    def print_submodules(self):
        print(f"Linear module of shape {(self.out_features, self.in_features)} and mass {self.mass}.")


class MultiHeadedLinear(Module):
    def __init__(self, out_features, in_features, num_heads, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1

        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x, w):
        return self.scale * torch.einsum('ijh, ...jh -> ...ih', w[0], x)

    def initialize(self, device):
        weight = torch.empty((self.out_features, self.in_features, self.num_heads), device=device, requires_grad=True)
        for head in range(self.num_heads):
            torch.nn.init.orthogonal_(weight[:,:,head])
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, w, target_norm):
        weight = w[0]
        if not hasattr(self, "u"):
            self.u = torch.randn_like(weight[0])
        v = torch.einsum('ab..., b... -> a...', weight, self.u)
        v /= v.norm(dim=0, keepdim=True)
        self.u = torch.einsum('a..., ab... -> b...', v, weight)
        weight *= target_norm / self.u.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def regularize(self, w, strength):
        weight = w[0]
        weight *= 1 - strength

    def print_submodules(self):
        print(f"Linear module of shape {(self.out_features, self.in_features)} with {self.num_heads} heads and mass {self.mass}.")


class Conv2D(Module):
    def __init__(self, out_channels, in_channels, kernel_size=3, stride=1, padding=1, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.k = kernel_size
        self.stride = stride
        self.pad = padding
        self.scale = math.sqrt(out_channels / in_channels) / (kernel_size ** 2)

    def forward(self, x, w):
        return self.scale * torch.nn.functional.conv2d(x, w[0], None, self.stride, self.pad)

    def initialize(self, device):
        weight = torch.empty((self.out_channels, self.in_channels, self.k, self.k), device=device, requires_grad=True)
        for kx in range(self.k):
            for ky in range(self.k):
                torch.nn.init.orthogonal_(weight[:,:,kx,ky])
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, w, target_norm):
        weight = w[0]
        if not hasattr(self, "u"):
            self.u = torch.randn_like(weight[0])
        v = torch.einsum('ab..., b... -> a...', weight, self.u)
        v /= v.norm(dim=0, keepdim=True)
        self.u = torch.einsum('a..., ab... -> b...', v, weight)
        weight *= target_norm / self.u.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def regularize(self, w, strength):
        weight = w[0]
        weight *= 1 - strength

    def print_submodules(self):
        print(f"Conv2D module of shape {(self.out_features, self.in_features, self.k, self.k)} and mass {self.mass}.")


class Embedding(Module):

    def __init__(self, num_embedding, embedding_dim, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1

        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.scale = math.sqrt(embedding_dim)

    def forward(self, x, w):
        return self.scale * torch.nn.functional.embedding(x, w[0])

    def initialize(self, device):
        weight = torch.empty((self.num_embedding, self.embedding_dim), device=device, requires_grad=True)
        torch.nn.init.normal_(weight)
        weight.data /= weight.norm(dim=1, keepdim=True)
        return Vector(weight)

    @torch.no_grad()
    def normalize(self, w, target_norm):
        weight = w[0]
        weight *= (target_norm / weight.norm(dim=1, keepdim=True)).nan_to_num_()

    @torch.no_grad()
    def regularize(self, w, strength):
        weight = w[0]
        weight /= weight.norm(dim=1, keepdim=True)

    def print_submodules(self):
        print(f"Embedding module: {self.num_embedding} embeddings of size {self.embedding_dim}. Mass {self.mass}.")
