import torch

from module.atomic import Linear
from module.bond   import Identity, ReLU, Abs, MeanSubtract, RMSDivide
from module.vector import cosine_similarity

# set the device
device = "cpu"

# optimisation hyperparameters
beta1 = 0.9
beta2 = 0.99
lr = 0.001
wd = 0.01

# set the network size
input_dim = 1000
output_dim = 10
num_blocks = 5
block_depth = 2
width = 100

# sample some fake training data
data   = torch.randn(32, input_dim,  device=device)
target = torch.randn(32, output_dim, device=device)

# define the network architecture
residue = (MeanSubtract() @ Abs() @ Linear(width, width) @ RMSDivide()) ** block_depth
block = (1-1/num_blocks)*Identity() + 1/num_blocks * residue
blocks = block ** num_blocks
blocks.tare()
net = Linear(output_dim, width) @ blocks @ Linear(width, input_dim)

# initialise the weights and optimisation state
weights = net.initialize(device=device)
mom1 = mom2 = old_grad = 0 * weights

# train forever
for step, _ in enumerate(iter(lambda:0,1)):

    loss = (target - net(data, weights)).square().mean()
    loss.backward()

    with torch.no_grad():
        lr *= 1 + cosine_similarity(old_grad, old_grad := weights.grad())

        mom1 += (1-beta1) * (weights.grad()    - mom1)
        mom2 += (1-beta2) * (weights.grad()**2 - mom2)

        weights -= lr * net.normalize(mom1 / mom2 ** 0.5)
        weights -= lr * wd * weights

    if step % 50 == 0:
        print(step, '\t', f"{loss.item():.4}", '\t', f"{lr:.4}")
