import math
import torch

from modula.abstract import Module
from modula.vector import Vector


class Bond(Module):
    """A module with no weights."""
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.length = 0
        self.initialize = lambda device, dtype : Vector()
        self.normalize  = lambda w, target_norm : None
        self.regularize = lambda w, strength : None


class Identity(Bond):
    """Identity module."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x


class Flatten(Bond):
    """Flatten all non-batch dimensions."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.flatten(start_dim=1)

# activation

class Abs(Bond):
    """Absolute value nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.abs(x)


class ReLU(Bond):
    """ReLU nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.relu(x)


def ScaledReLU():
    """ReLU scaled to have sensitivity one."""
    return math.sqrt(2) * ReLU()


class GELU(Bond):
    """GELU nonlinearity."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: torch.nn.functional.gelu(x)


def ScaledGELU():
    """GELU scaled to have sensitivity 1."""
    return math.sqrt(2) * GELU()


# pooling

class AvgPool2D(Bond):
    """Average pooling that adapts to different input sizes."""
    def __init__(self, output_size = (1,1)):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.nn.functional.adaptive_avg_pool2d(x, output_size)

class AvgPool1D(Bond):
    """Average pooling that adapts to different input sizes."""
    def __init__(self, output_size = 1):
        super().__init__()
        self.sensitivity = 1
        self.output_size = output_size
        #self.forward = lambda x, w: torch.nn.functional.adaptive_avg_pool1d(x, output_size)

    def forward(self, x, w):
        #print(x.shape)
        out = torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)
        #print('pooled shape:', out.shape)
        return out


# transformer utilities

class AddHeads(Bond):
    """Reshapes an input to have heads.

    Input shape: batch_size, sequence_length, embed_dim
    Output shape: batch_size, num_heads, sequence_length, head_size

    Adapted from Karpathy's nanoGPT.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.sensitivity = 1
        self.num_heads = num_heads

    def forward(self, x, w):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


class RemoveHeads(Bond):
    """Inverse of AddHeads."""
    def __init__(self):
        super().__init__()
        self.sensitivity = 1

    def forward(self, x, w):
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh*hs)

class RoPE(Bond):
    """Rotary embedding
    """
    def __init__(self, context, dim, base=10000):
        super().__init__()
        self.sensitivity = 1
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = context
        self.base = base
        self.dim = dim

        # precompute cos_cached, sin_cached in fp32
        cos_cached, sin_cached = self._prepare_cache(context, base)

        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, base):
        # precompute cos_cached, sin_cached in fp32
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        return (
            cos_cached,
            sin_cached
        )

    def forward(self, x, w):
        B, H, T, D = x.size()
        assert T <= self.max_seq_len
        # prepare rope cos and sin
        if T != self.max_seq_len:
            cos = self.cos_cached[:T, ...].to(x.device)
            sin = self.sin_cached[:T, ...].to(x.device)

        else:
            cos = self.cos_cached.to(x.device)
            sin = self.sin_cached.to(x.device)

        assert self.dim <= D
        # split channels, apply rope to first self.dim channels 
        if self.dim < D:
            x_rot, x_pass = (x[..., : self.dim], x[..., self.dim:])
        else:
            x_rot = x
        # split the first self.bim channels in half
        # note this is different from llama's implementation with view_as_complex
        x1, x2 = x_rot[..., : x_rot.shape[-1] // 2], x_rot[..., x_rot.shape[-1] // 2 :]
        x3 = torch.cat((-x2, x1), dim=x1.ndim - 1)
        # apply RoPE
        x_rot = x_rot * cos + x3 * sin

        if self.dim < D:
            return torch.cat((x_rot, x_pass), dim=-1)
        else:
            return x_rot
        

class Enumerate(Bond):
    """Replace each column with its column index. Used to make position embeddings."""
    def __init__(self, num = None):
        super().__init__()
        self.sensitivity = 1
        self.num = num

    def forward(self, x, w):
        ul =  x.size(1) if not self.num else self.num
        out = torch.arange(0, ul, dtype=torch.long, device=x.device)
        return out



class MeanSubtract(Bond):
    """Mean subtraction."""
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x - x.mean(dim=dim, keepdim=True)


class RMSDivide(Bond):
    """Normalize to have unit RMS norm."""
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x / x.square().mean(dim=dim, keepdim=True).sqrt()


def LayerNorm(dim=-1):
    """Mean subtraction followed by RMS normalization."""
    return RMSDivide(dim) @ MeanSubtract(dim)


class Mean(Bond):
    """Take the mean over a specified dimension."""
    def __init__(self, dim):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.mean(dim=dim)


class Permute(Bond):
    """(b, n, d) -> (b, d, n)"""
    def __init__(self, permute=(0,2,1)):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x.permute(*permute)


class FunctionalAttention(Bond):
    """The part of attention that doesn't involve weights."""

    def __init__(self, causal):
        super().__init__()
        self.sensitivity = 1
        self.causal = causal

    def forward(self, x, w):
        q, k, v = x

        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=1/q.shape[-1])


class PatchifyImage(Bond):
    """Patchify images. (b, 3, h, w) -> (b, n, f)"""
    def __init__(self, patch_size = 8, image_size = 32):
        super().__init__()
        self.sensitivity = 1
        self.patch_size  = patch_size
        self.n_patch  = image_size // patch_size
    def forward(self, x, w):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.n_patch**2 ,-1)
        return out


class AddCLS(Bond):
    """Adds CLS token(s) via concatination. (b, n, w) -> (b, n+len(cls), d)"""
    def __init__(self, dim = 1):
        super().__init__()
        self.sensitivity = 1/2
        self.dim = dim

    def forward(self, x, w):
        cls, tokens = x
        out = torch.cat([cls.repeat(tokens.size(0),1,1), tokens],dim=self.dim)
        return out


class ExtractCLS(Bond):
    """
    Extract CLS token from (b, n, d) -> (b, len(indices), d)
    This allows for extracting multiple tokens/registers for downstream tasks
    """
    def __init__(self, indices=[0]):
        super().__init__()
        self.sensitivity = 1
        self.indices = indices

    def forward(self, x, w):
        out =x[:,self.indices]
        return out