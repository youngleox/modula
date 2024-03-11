import math
import torch
from tqdm.auto import trange

from data.dataset import getIterator
from module.compound import MLP, ResMLP

if __name__ == '__main__':

    steps = 1000
    batch_size = 100

    getBatch = getIterator(dataset="cifar10", batch_size=batch_size)
    net = ResMLP(width=1000, num_blocks=10, block_depth=1, input_dim=3072, output_dim=10)

    net.initialize()

    for i in (pbar := trange(steps)):
        data, target = getBatch(train = True)

        data = data.flatten(start_dim=1)

        onehot = torch.nn.functional.one_hot(target, num_classes=10).float()
        onehot *= math.sqrt(10)

        out = net.forward(data)
        loss = (out-onehot).square().mean()
        loss.backward()

        net.update(0.5 * (1 - i / steps), beta=0.9, wd=0.01)
        net.zero_grad()

        acc = (out.argmax(dim=1) == target).sum() / batch_size
        
        pbar.set_description(f"acc: {acc.item():.4f}")

    print(f"final loss {loss.item()}")