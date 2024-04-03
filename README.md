<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/modula.svg">
  <source media="(prefers-color-scheme: light)" srcset="logo/modula_light.svg">
  <img alt="modula logo" src="logo/modula.svg">
</picture>

`Modula` is a deep learning framework designed for graceful scaling of training. In Modula, the user defines a network architecture, by arbitrarily combining and composing `atomic modules`. Modula then automatically constructs an optimizer tailored specifically to this network architecture. We call this optimizer `modular gradient descent`. Modular gradient descent is built such that its optimal learning rate remains roughly fixed as the number and size of atomic modules is scaled. Essentially, Modula automates the computation of architecture-optimizer scaling rules like maximal update parameterization (μP) for any computation graph.

## Example

```python
from torch import randn
from modula.atomic import Identity, Linear, ReLU, MeanSubtract, RMSNorm

# sample some fake training data
data = randn(32,1000)
target = randn(32,10)

# set the network size
num_blocks = 10
block_depth = 3
width = 100

# define the network architecture
residue = (MeanSubtract() @ ReLU() @ Linear(width, width) @ RMSNorm()) ** block_depth
blocks = (Identity() + 1/num_blocks * residue) ** num_blocks
net = Linear(10, width) @ blocks @ Linear(width, 1000)

# run the training
for _ in range(steps := 1000):
    loss = (y - net(x)).square().mean()
    loss.backward()
    net.update(lr = 0.5, hps = {'beta':0.9, 'wd':0.01})
```
