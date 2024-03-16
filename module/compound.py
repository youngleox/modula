from module.atomic import Identity, Linear, ScaledReLU, MeanSubtract, Abs, Conv2D, Flatten


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


def CNN(width, depth, input_dim, output_dim, num_pixels):
	initial = Conv2D(width, input_dim)
	block = ScaledReLU() @ Conv2D(width, width)
	downscale = Conv2D(1, width)
	final = Linear(output_dim, num_pixels)

	net = final @ Flatten() @ downscale @ block ** depth @ initial

	return net


def ResCNN(width, num_blocks, block_depth, input_dim, output_dim, num_pixels):
	initial = Conv2D(width, input_dim)

	residue = (MeanSubtract() @ Abs() @ Conv2D(width, width)) ** block_depth
	block = (1-1/num_blocks) * Identity() + 1/num_blocks * residue

	downscale = Conv2D(1, width)
	final = Linear(output_dim, num_pixels)

	net = final @ Flatten() @ downscale @ block ** num_blocks @ initial

	return net
