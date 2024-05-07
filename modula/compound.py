import math
from modula.atom import *
from modula.bond import *


def ResMLP(width, num_blocks, block_depth, input_dim, output_dim):
    initial = Linear(width, math.prod(input_dim)) @ Flatten()

    residue = MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide()
    block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue ** block_depth
    blocks = block ** num_blocks
    blocks.tare()

    final = Linear(output_dim, width)

    return final @ blocks @ initial


def ResCNN(width, num_blocks, block_depth, input_dim, output_dim):
    initial = Conv2D(width, input_dim[0])

    residue = MeanSubtract(dim=(1,2,3)) @ Abs() @ Conv2D(width, width) @ RMSDivide(dim=(1,2,3))
    block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue ** block_depth
    blocks = block ** num_blocks
    blocks.tare()

    final = Linear(output_dim, width) @ Flatten() @ AvgPool()

    return final @ blocks @ initial


def Attention(num_heads, d_embed, d_query, d_value, context, causal):
    Q = MultiHeadedLinear(d_query, d_embed, num_heads)
    K = MultiHeadedLinear(d_query, d_embed, num_heads)
    V = MultiHeadedLinear(d_value, d_embed, num_heads)
    W = MultiHeadedLinear(d_embed, d_value, num_heads)

    return Mean(dim=-1) @ W @ FunctionalAttention(causal) * 1/3 @ (Q, K, V) @ Duplicate(num_heads)


def GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks):
    token_embedding = Embedding(vocab_size, d_embed)
    position_embedding = Embedding(context, d_embed) @ Enumerate()
    initial = 1/2 * token_embedding + 1/2 * position_embedding
    initial.tare()

    attention = Attention(num_heads, d_embed, d_query, d_value, context, causal=True) @ LayerNorm()
    mlp = ScaledGELU() @ Linear(d_embed, 4*d_embed) @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * attention
    mlp_block       = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare()

    final = Linear(vocab_size, d_embed) @ LayerNorm()

    return final @ blocks @ initial
