import math
import torch

from module.abstract import Module

def spectral_norm(p, u, v, num_steps=1, eps=1e-12):
    if not p.dim() > 1:
        return p.norm(), u, v
    for _ in range(num_steps):
        v = torch.nn.functional.normalize(torch.mv(p.t(), u), dim=0, eps=eps, out=v)
        u = torch.nn.functional.normalize(torch.mv(p, v), dim=0, eps=eps, out=u)
    sigma = torch.dot(u, torch.mv(p, v))
    return sigma, u, v

class SpectralNormalizer:
    def __init__(self, target_norm, mode='linear', use_cache=1):
        assert mode in ['linear', 'conv2d', 'embedding']
        self.mode = mode
        self.target_norm = target_norm
        self.use_cache = use_cache
        self.u, self.v = None, None

    def normalize(self, w):
        if self.mode == 'linear':
            if self.u is None and self.use_cache:
                self.u = torch.randn_like(w[:,0])
                self.v = torch.randn_like(w[0])
            norm, self.u.data, self.v.data = spectral_norm(w, self.u, self.v)
            return w / norm * self.target_norm
        
        elif self.mode == 'embedding':
            return w[0] / w[0].norm(dim=1, keepdim=True) * self.target_norm

class Linear(Module):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1

        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

        self.normalizer = None

    def forward(self, x, w):
        return self.scale * torch.nn.functional.linear(x, w[0])

    def initialize(self, device):
        weight = torch.empty((self.out_features, self.in_features), device=device, requires_grad=True)
        torch.nn.init.orthogonal_(weight)
        return [weight]

    def initialize_normalizer(self, target_norm):
        if self.normalizer is None:
            self.normalizer = SpectralNormalizer(target_norm=target_norm)
        return [self.normalizer]
    
    @torch.no_grad()
    def normalize(self, w, target_norm):
        return [w[0] / spectral_norm(w[0]) * target_norm]

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
        return [weight]

    @torch.no_grad()
    def normalize(self, w, target_norm):
        return [w[0] / spectral_norm(w[0]) * target_norm]

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
        return [weight]

    @torch.no_grad()
    def normalize(self, w, target_norm):
        return [w[0] / spectral_norm(w[0]) * target_norm]

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
        weight.data /= self.weight.norm(dim=1, keepdim=True)
        return [weight]

    @torch.no_grad()
    def normalize(self, w, target_norm):
        return [w[0] / w[0].norm(dim=1, keepdim=True) * target_norm]

    def print_submodules(self):
        print(f"Embedding module: {self.num_embedding} embeddings of size {self.embedding_dim}. Mass {self.mass}.")
