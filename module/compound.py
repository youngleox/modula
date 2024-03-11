from module.atomic import Identity, Linear, ScaledReLU


def MLP(width, depth, input_dim, output_dim):
	block = ScaledReLU() @ Linear(width, width)

	net = ScaledReLU() @ Linear(width, input_dim)
	net = block ** (depth-2) @ net
	net = Linear(output_dim, width) @ net

	return net


def ResMLP(width, num_blocks, block_depth, input_dim, output_dim):
	residue = (Linear(width, width) @ ScaledReLU()) ** block_depth
	block = (1-1/depth) * Identity() + 1/depth * residue

	net = ScaledReLU() @ Linear(width, input_dim)
	net = block ** num_blocks @ net
	net = Linear(output_dim, width) @ net


