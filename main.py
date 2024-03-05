import math
import torch

from module.atomics import Identity, Linear, ReLU

from tqdm.auto import trange

steps = 1000
width = 1024
depth = 10

init_lr = 0.1
beta = 0.9
wd = 0.01

x = torch.randn(32, 8)
y = torch.randn(32, 8)

block = (1-1/depth) * Identity() + 1/depth * Linear(width, width) @ (math.sqrt(2) * ReLU())
# block = Linear(width, width) @ (math.sqrt(2) * ReLU())
net = Linear(8,width) @ block ** depth @ Linear(width, 8)
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