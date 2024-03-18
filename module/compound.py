from module.atomic import *


def MLP(width, depth, input_dim, output_dim):
	block = ScaledReLU() @ Linear(width, width)

	net = ScaledReLU() @ Linear(width, input_dim)
	net = block ** (depth-2) @ net
	net = Linear(output_dim, width) @ net

	return net


def ResMLP(width, num_blocks, block_depth, input_dim, output_dim):
	residue = (MeanSubtract() @ Abs() @ Linear(width, width)) ** block_depth
	block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue

	net = Linear(width, input_dim)
	net = block ** num_blocks @ net
	net = Linear(output_dim, width) @ net

	return net


def ResCNN(width, num_blocks, block_depth, input_dim, output_dim, num_pixels):
	net = Conv2D(width, input_dim)

	residue = (MeanSubtract() @ Abs() @ Conv2D(width, width) @ LayerNorm()) ** block_depth
	block = 1/2 * Identity() + 1/2 * residue
	net = block**2 @ net

	for _ in range(3):
		residue = MeanSubtract() @ Abs() @ Conv2D(2*width, width, stride = 2)  @ LayerNorm()
		residue = MeanSubtract() @ Abs() @ Conv2D(2*width, 2*width) @ LayerNorm() @ residue
		skip = Conv2D(2*width, width, kernel_size = 1, stride = 2, padding = 0)  @ LayerNorm()
		block_1 = 1/2 * skip + 1/2 * residue

		residue = (MeanSubtract() @ Abs() @ Conv2D(2*width, 2*width) @ LayerNorm())**2
		block_2 = 1/2 * Identity() + 1/2 * residue

		net = block_2 @ block_1 @ net
		width *= 2

	final = Linear(output_dim, width) @ Flatten() @ AvgPool()
	net = final @ net

	return net
