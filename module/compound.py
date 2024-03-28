from module.atomic import *
import numpy


def residualize(residue, num_blocks, block_depth):
	assert num_blocks > 1, "need at least two blocks"

	block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue ** block_depth

	return block ** num_blocks


def ResMLP(width, num_blocks, block_depth, input_dim, output_dim):
	initial = Linear(width, numpy.prod(input_dim)) @ Flatten()

	residue = MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide()
	blocks = residualize(residue, num_blocks, block_depth)
	blocks.tare()

	final = Linear(output_dim, width)

	return final @ blocks @ initial


def ResCNN(width, num_blocks, block_depth, input_dim, output_dim):
	initial = Conv2D(width, input_dim[0])

	residue = MeanSubtract() @ Abs() @ Conv2D(width, width) @ RMSDivide()
	blocks = residualize(residue, num_blocks, block_depth)
	blocks.tare()

	final = Linear(output_dim, width) @ Flatten() @ AvgPool()

	return final @ blocks @ initial


def Attention(num_heads, d_embed, d_query, d_value, context, causal, mass=4):
    Q = MultiHeadedLinear(d_query, d_embed, num_heads, mass=mass/4)
    K = MultiHeadedLinear(d_query, d_embed, num_heads, mass=mass/4)
    V = MultiHeadedLinear(d_value, d_embed, num_heads, mass=mass/4)
    W = MultiHeadedLinear(d_embed, d_value, num_heads, mass=mass/4)

    output = Duplicate(num_heads)
    output = TupleModule((Q, K, V)) @ output
    output = FunctionalAttention(context, causal) @ output
    output = W @ output
    output = Mean(dim=-1) @ output

    return output


def GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks):
	token_embedding = Embedding(vocab_size, d_embed)
	position_embedding = Embedding(context, d_embed) @ Enumerate()
	initial = 1/2 * token_embedding + 1/2 * position_embedding

	mlp = Linear(d_embed, 4*d_embed) @ MeanSubtract() @ Abs() @ Linear(4*d_embed, d_embed)
	attention = Attention(num_heads, d_embed, d_query, d_value, context, causal=True)
	residue = MeanSubtract() @ (1/2 * mlp + 1/2 * attention) @ RMSDivide()
	blocks = residualize(residue, num_blocks, block_depth=1)
	blocks.tare()

	final = Linear(vocab_size, d_embed) @ RMSDivide()

	return final @ blocks @ initial
