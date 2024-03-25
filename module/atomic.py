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


class Enumerate(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: torch.arange(0, x.size()[1], dtype=torch.long, device=x.device)
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
    def update(self, lr, hps):
        self.momentum += (1-hps["beta"]) * (self.weight.grad - self.momentum)
        norm, self.u.data = spectral_norm(self.momentum, self.u)

        if norm == 0.0:
            self.u = torch.randn_like(self.weight[0])
        else:
            self.weight -= lr * self.momentum / norm
            self.weight *= 1 - lr * hps["wd"]

        self.weight.grad = None

    def print_submodules(self):
        print(f"Linear module of shape {(self.out_features, self.in_features)} and mass {self.mass}.")


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

    def forward(self, input):
        return self.scale * torch.nn.functional.embedding(input, self.weight)

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


class Attention(Module):

    def __init__(self, num_heads, d_embed, d_query, d_value, context, causal, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1

        self.num_heads = num_heads
        self.d_embed = d_embed
        self.d_query = d_query
        self.d_value = d_value
        self.context = context
        self.causal = causal

    def forward(self, x):
        ''' For einsum adopt notation:
                b = batch_size
                h = num_heads
                q = d_query
                v = d_value
                d = d_embed
                c = context
        '''
        q = torch.einsum('qdh, bcd -> bhcq', self.Q, x) * math.sqrt(self.d_query / self.d_embed)
        k = torch.einsum('qdh, bcd -> bhcq', self.K, x) * math.sqrt(self.d_query / self.d_embed)
        v = torch.einsum('vdh, bcd -> bhcv', self.V, x) * math.sqrt(self.d_value / self.d_embed)

        # TODO: check scaling factor (/ self.d_query)
        att = torch.einsum('bhcq, bhCq -> bhcC', q, k) / self.d_query
        if self.causal:
            T = x.size()[1]
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        p = torch.nn.functional.softmax(att, dim=-1)
        y = torch.einsum('bhcC, bhCv -> bhcv', p, v)
        z = torch.einsum('dvh, bhcv -> bcd', self.W, y) * math.sqrt(self.d_embed / self.d_value) / self.num_heads

        # TODO: check contiguous memory

        return z

    def initialize(self, device):
        self.Q = torch.empty((self.d_query, self.d_embed, self.num_heads), device=device, requires_grad=True)
        self.K = torch.empty((self.d_query, self.d_embed, self.num_heads), device=device, requires_grad=True)
        self.V = torch.empty((self.d_value, self.d_embed, self.num_heads), device=device, requires_grad=True)
        self.W = torch.empty((self.d_embed, self.d_value, self.num_heads), device=device, requires_grad=True)

        self.parameters = [self.Q, self.K, self.V, self.W]

        for M in self.parameters:
            for head in range(self.num_heads):
                torch.nn.init.orthogonal_(M[:,:,head])

        if self.causal:
            T = self.context
            self.mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)

        self.us = [torch.randn_like(M[0]) for M in self.parameters]

    @torch.no_grad()
    def update(self, lr, hps):
        for M, u in zip(self.parameters, self.us):
            norm, u.data = spectral_norm(M.grad, u)

            if (norm == 0.0).any():
                u.data = torch.randn_like(M[0])
            else:
                M -= lr / 4 * M.grad / norm
                M *= 1 - lr * hps["wd"]

            M.grad = None

    def print_submodules(self):
        print(f"Attention module. Mass {self.mass}.")
