import math
import torch

from module.compound import MLP, ResMLP

from tqdm.auto import trange

steps = 1000

width = 1024
num_blocks = 10

block_depth = 2
input_dim = 8
output_dim = 8
batch_size = 32

init_lr = 0.5
beta = 0.9
wd = 0.01

x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)

net = ResMLP(width, num_blocks, block_depth, input_dim, output_dim)
print(net)

net.initialize()

# net = net.cuda()
# x = x.cuda()
# y = y.cuda()

for i in (pbar := trange(steps)):
    out = net.forward(x)
    loss = (out-y).square().mean()
    loss.backward()

    net.update(init_lr * (1 - i / steps), beta, wd)
    net.zero_grad()
        
    pbar.set_description(f"loss: {loss.item():.4f}")

print(f"final loss {loss.item()}")