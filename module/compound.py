from module.atomic import *
import numpy


def residualize(residue, num_blocks, block_depth):
	assert args.num_blocks > 1, "need at least two blocks"

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
