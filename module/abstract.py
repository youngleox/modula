import copy


class Module:
    def __init__(self):
        self.mass = None
        self.sensitivity = None
        self.parameters = []
        self.children = []
        
    def forward(self, x):
        raise NotImplementedError

    def initialize(self, device):
        pass

    def update(self, lr, hps):
        pass

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative = relative)
        else:
            self.tare(relative = absolute / self.mass)

    def print_submodules(self):
        for child in self.children:
            child.print_submodules()

    def __str__(self):
        return f"Module of mass {self.mass} and sensitivity {self.sensitivity}."

    def __call__(self, x):
        return self.forward(x)

    def __matmul__(self, other):
        if isinstance(other, tuple): other = TupleModule(other)
        return CompositeModule(self, other)

    def __rmatmul__(self, other):
        if isinstance(other, tuple): other = TupleModule(other)
        return other @ self

    def __add__(self, other):
        return Add() @ (self, other)

    def __mul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return self @ ScalarMultiply(other)

    def __rmul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return ScalarMultiply(other) @ self

    def __truediv__(self, other):
        assert other != 0, "cannot divide a module by zero"
        return self * (1/other)

    def __pow__(self, other):
        assert other >= 0 and other % 1 == 0, "nonnegative integer powers only"
        if other > 0:
            return self @ copy.deepcopy(self) ** (other - 1)
        else:
            return ScalarMultiply(1.0)


class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)

        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity
        
    def forward(self, x):
        m0, m1 = self.children
        return m1.forward(m0.forward(x))

    def initialize(self, device):
        m0, m1 = self.children
        m0.initialize(device)
        m1.initialize(device)
        self.parameters = m0.parameters + m1.parameters

    def update(self, lr, hps):
        m0, m1 = self.children
        if self.mass > 0:
            m0.update(m0.mass / self.mass / m1.sensitivity * lr, hps)
            m1.update(m1.mass / self.mass                  * lr, hps)


class TupleModule(Module):
    def __init__(self, tuple_of_modules):
        super().__init__()
        self.children = tuple_of_modules

        self.mass        = sum(child.mass        for child in self.children)
        self.sensitivity = sum(child.sensitivity for child in self.children)
        
    def forward(self, x):
        return tuple(child.forward(x) for child in self.children)

    def initialize(self, device):
        for child in self.children:
            child.initialize(device)
            self.parameters += child.parameters

    def update(self, lr, hps):
        if self.mass > 0:
            for child in self.children:
                child.update(child.mass / self.mass * lr, hps)


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)

        self.alpha = alpha

    def forward(self, x):
        if isinstance(x, tuple):
            return tuple(self.forward(x_i) for x_i in x)
        else:
            return self.alpha*x

class Add(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1

    def forward(self, x):
        assert isinstance(x, tuple), "can only compose add with tuples"
        return sum(x_i for x_i in x)
