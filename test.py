import torch
import math

from module.atomic import Linear
from module.bond import Identity, ReLU, MeanSubtract, RMSDivide
from module.vector import cosine_similarity

# set the device
device = "cuda"

# optimisation hyperparameters
steps = 1000
beta1 = 0.9
beta2 = 0.99
lr = 0.0001

# set the network size
input_dim = 1000
output_dim = 10
num_blocks = 5
block_depth = 3
width = 100

# sample some fake training data
data = torch.randn(3200, input_dim, device=device)
target = torch.randn(3200, output_dim, device=device)

# define the network architecture
residue = (MeanSubtract() @ ReLU() @ Linear(width, width) @ RMSDivide()) ** block_depth
blocks = (Identity() + 1/num_blocks * residue) ** num_blocks
net = Linear(output_dim, width) @ blocks @ Linear(width, input_dim)

# initialise the weights and optimisation state
init_weights = weights = net.initialize(device=device)
mom1 = 0 * weights
mom2 = 0 * weights

# run the training
for step in range(steps):
    loss = (target - net(data, weights)).square().mean()
    loss.backward()

    with torch.no_grad():
        lr *= 1 + cosine_similarity(init_weights - weights, weights.grad())

        mom1 += (1-beta1) * (weights.grad()    - mom1)
        mom2 += (1-beta2) * (weights.grad()**2 - mom2)

        weights -= lr * (1-step/steps) * net.normalize(mom1 / mom2 ** 0.5)

    weights.zero_grad()

    if step % 50 == 0:
        print(step, '\t', round(loss.item(),9), '\t', round(lr,5))
