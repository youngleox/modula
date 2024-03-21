from module.atomic import *
import numpy


def MLP(args, input_dim, output_dim):
	input_dim = numpy.prod(input_dim)
	width = args.width
	depth = args.depth

	block = ScaledReLU() @ Linear(width, width)

	net = ScaledReLU() @ Linear(width, input_dim) @ Flatten()
	net = block ** (depth-2) @ net
	net = Linear(output_dim, width) @ net

	return net


def ResMLP(args, input_dim, output_dim):
	input_dim = numpy.prod(input_dim)
	width = args.width
	num_blocks = args.depth
	block_depth = args.blockdepth

	residue = (MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide()) ** block_depth
	block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue
	blocks = block ** num_blocks
	blocks.tare()

	return Linear(output_dim, width) @ blocks @ Linear(width, input_dim) @ Flatten()

def ResCNN(args, input_dim, output_dim):
	width = args.width
	blockdepth = args.blockdepth
	num_blocks = args.depth
	input_dim = input_dim[0]

	assert args.depth > 1, "need at least two blocks"

	Layer      = lambda width : MeanSubtract() @ Abs() @ Conv2D(width, width)                     @ LayerNorm()
	DownSample = lambda width : MeanSubtract() @ Abs() @ Conv2D(2*width, width, stride = 2)       @ LayerNorm()
	Skip       = lambda width : Conv2D(2*width, width, kernel_size = 1, stride = 2, padding = 0)  @ LayerNorm()

	net = Conv2D(width, input_dim)

	block = 1/2 * Identity() + 1/2 * Layer(width)**blockdepth
	net = block ** 2 @ net

	for _ in range(3):
		block_1 = 1/2 * Skip(width) + 1/2 * Layer(2*width)**(blockdepth-1) @ DownSample(width)
		block_2 = 1/2 * Identity()  + 1/2 * Layer(2*width)**blockdepth
		net = block_2 @ block_1 @ net

		width *= 2

	block = (1-1/num_blocks) * Identity() + 1/num_blocks * Layer(width)**blockdepth
	net = block ** num_blocks @ net

	final = Linear(output_dim, width) @ Flatten() @ AvgPool()
	net = final @ net

	return net
