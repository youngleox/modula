import torch 
from modula.atom import *
from modula.bond import *


def ResMLP(width, num_blocks, block_depth, input_dim, output_dim, body_mass=1.0):
    """ResNet where the blocks are MLPs."""
    initial = Linear(width, input_dim) @ Flatten()

    residue = MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide()
    block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue ** block_depth
    blocks = block ** num_blocks
    if body_mass > 0.0:
        blocks.tare(absolute=body_mass)

    final = Linear(output_dim, width) @ Linear(width,width)

    return final @ blocks @ initial

def ResMLP2(input_dim, output_dim,
            width, num_blocks, 
            block_depth, initial_depth=1, final_depth=1,
            body_mass=1.0, initial_mass=1.0, final_mass=1.0,
            layer='naw'):
    """ResNet where the blocks are MLPs."""
    # 
    assert initial_depth >= 1 and final_depth >= 1
    if layer == 'j':
        layer = MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide() 
    elif layer == 'naw':
        layer = Linear(width, width) @ ScaledReLU() @ LayerNorm()
    elif layer == 'nwa':
        layer = ScaledReLU() @ Linear(width, width) @ LayerNorm()
    elif layer == 'wa':
        layer = ScaledReLU() @ Linear(width, width)
    elif layer == 'aw':
        layer = Linear(width, width) @ ScaledReLU()
    else:
        print('Not implemented')

    # define intial layers
    neck = layer ** (initial_depth - 1)  if initial_depth > 1 else Identity()
       
    initial = neck @ Linear(width, input_dim) @ Flatten()
    if initial_mass > 0.0:
        initial.tare(absolute=initial_mass)
    # define body
    block = (1-1/num_blocks) * Identity() + 1/num_blocks * layer ** block_depth
    blocks = block ** num_blocks
    if body_mass > 0.0:
        blocks.tare(absolute=body_mass)
    # define final layers
    tail = layer ** (final_depth - 1) if final_depth > 1 else Identity()
    final = Linear(output_dim, width) @ tail
    if final_mass > 0.0:
        final.tare(absolute=final_mass)

    return final @ blocks @ initial


def ResCNN(width, num_blocks, block_depth, input_dim, output_dim):
    """ResNet where the blocks are CNNs."""
    initial = Conv2D(width, input_dim)

    residue = MeanSubtract(dim=(1,2,3)) @ Abs() @ Conv2D(width, width) @ RMSDivide(dim=(1,2,3))
    block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue ** block_depth
    blocks = block ** num_blocks
    blocks.tare()

    final = Linear(output_dim, width) @ Flatten() @ AvgPool2D()

    return final @ blocks @ initial


def Attention(num_heads, d_embed, d_query, d_value, context, causal):
    """Multi-head attention."""
    Q = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = AddHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, d_value * num_heads) @ RemoveHeads()

    return W @ FunctionalAttention(causal) * 1/3 @ (Q, K, V)


def GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks, body_mass=5):
    """GPT."""
    token_embedding = Embedding(vocab_size, d_embed)
    position_embedding = Embedding(context, d_embed) @ Enumerate()
    initial = 1/2 * token_embedding + 1/2 * position_embedding
    initial.tare()

    attention = Attention(num_heads, d_embed, d_query, d_value, context, causal=True) @ LayerNorm()
    mlp = Linear(d_embed, 4*d_embed) @ ScaledGELU() @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * attention
    mlp_block       = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare(absolute=body_mass)

    final = Linear(vocab_size, d_embed) @ LayerNorm()

    return final @ blocks @ initial

def Attention_with_RoPE(num_heads, d_embed, d_query, d_value, context, causal, d_rope):
    """Multi-head attention."""
    Q = RoPE(context, d_rope) @ AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = RoPE(context, d_rope) @ AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = AddHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, d_value * num_heads) @ RemoveHeads()
    print(Q)
    return W @ FunctionalAttention(causal) * 1/3 @ (Q, K, V)

def Pythia(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks, body_mass=5):
    """GPT."""
    token_embedding = Embedding(vocab_size, d_embed)
    initial = token_embedding
    initial.tare()
    a = 0.5
    attention = Attention_with_RoPE(num_heads, d_embed, d_query, d_value, context, causal=True, d_rope=int(0.25*d_query)) @ LayerNorm()
    mlp = Linear(d_embed, 4*d_embed) @ ScaledGELU() @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(num_blocks)) * Identity() + a/(num_blocks) * attention + (1-a)/(num_blocks) * mlp

    blocks = (attention_block) ** num_blocks
    blocks.tare(absolute=body_mass)

    final = Linear(vocab_size, d_embed) @ LayerNorm()

    return final @ blocks @ initial

def ViT(image_size, patch_size, output_dim, num_heads, d_embed, d_query, d_value, num_blocks, body_mass=5):
    """ViT."""

    n_token = (image_size // patch_size) ** 2
    d_patch = (patch_size)**2*3
    img_patch = PatchifyImage(patch_size,image_size)
    img_token = Linear(d_embed,d_patch) @ img_patch
    cls_token = Embedding(1, d_embed) @ Enumerate(1) @ img_patch
    token_embedding = AddCLS() @ (cls_token, img_token)
    position_embedding = Embedding(n_token+1, d_embed) @ Enumerate(n_token+1) @ img_patch
    initial = 1/2 * token_embedding + 1/2 * position_embedding
    initial.tare()

    attention = Attention(num_heads, d_embed, d_query, d_value, n_token+1, causal=False) @ LayerNorm()
    mlp = Linear(d_embed, 4*d_embed) @ ScaledGELU() @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * attention
    mlp_block       = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare(absolute=body_mass)

    final = Linear(output_dim, d_embed) @ LayerNorm() @ Flatten() @ Permute() @ ExtractCLS()

    return final @ blocks @ initial