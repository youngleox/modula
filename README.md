<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/modula.svg">
  <source media="(prefers-color-scheme: light)" srcset="logo/modula_light.svg">
  <img alt="modula logo" src="logo/modula.svg">
</picture>

Modula is a deep learning framework designed for graceful scaling. The user defines a network architecture in Modula by arbitrarily combining and composing atomic modules. Modula automatically constructs the modular gradient descent optimizer tailored to this network architecture. Modular gradient descent has the property that its optimal learning rate remains roughly fixed as the number and size of atomic modules is scaled. Essentially, Modula automates the computation of architecture-optimizer scaling rules for any computation graph. Modula is built on top of [PyTorch](https://pytorch.org/).

## Installation

local pip install:

```bash
pip install -e .
```

## Repository structure

```
.
├── examples
│   ├── hello-world.py              # simple training loop
│   ├── gradient-accumulation.py    # gradient accumulation for large batch training
│   └── multi-gpu.py                # multi GPU training with torch.distributed
├── logo
    └── ...
├── modula
│   ├── abstract.py                 # basic definitions: composition & concatenation, etc.
│   ├── atom.py                     # modules with weights: linear, conv2d etc.
│   ├── bond.py                     # modules without weights: ReLU, FunctionalAttention, etc.
│   ├── compound.py                 # derived modules: GPT, ResNet, etc.
│   └── vector.py                   # class for storing weight vectors
├── paper
    └── ...
├── README.md                       # this file
└── setup.py                        # pip package stuff
```

## Example

```python
from torch import randn, no_grad
from modula.atom import Linear
from modula.bond import ReLU

data, target = randn(1000), randn(10)

mlp = Linear(10,10000) @ ReLU() @ Linear(10000, 1000)
weights = mlp.initialize(device="cpu")

for _ in range(steps:=20):
    output = mlp(data, weights)

    loss = (target - output).square().mean()
    loss.backward()

    with no_grad():
        mlp.normalize(grad := weights.grad())
        weights -= 0.1 * grad
        weights.zero_grad()
    
        mlp.regularize(weights, strength = 0.01)

    print(_, loss.item())
```
