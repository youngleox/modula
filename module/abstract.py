import copy

from module.vector import Vector


class Module:
    def __init__(self):
        self.mass = None
        self.sensitivity = None
        self.children = []
        
    def forward(self, x, w):
        raise NotImplementedError

    def initialize(self, device):
        raise NotImplementedError

    def normalize(self, vector, target_norm):
        raise NotImplementedError

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

    def __call__(self, x, w):
        return self.forward(x, w)

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
            return copy.deepcopy(self) @ self ** (other - 1)
        else:
            return ScalarMultiply(1.0)


class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)

        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity
        
    def forward(self, x, w):
        m0, m1 = self.children
        w0, w1 = w
        return m1.forward(m0.forward(x, w0), w1)

    def initialize(self, device):
        return Vector(tuple(child.initialize(device) for child in self.children))

    def normalize(self, vector, target_norm=1):
        m0, m1 = self.children
        v0, v1 = vector

        if self.mass > 0:
            v0_normalized = m0.normalize(v0, target_norm=m0.mass / self.mass * target_norm / m1.sensitivity)
            v1_normalized = m1.normalize(v1, target_norm=m1.mass / self.mass * target_norm)
            return Vector((v0_normalized, v1_normalized))
        elif vector.length > 0:
            return vector * 0
        else:
            return vector


class TupleModule(Module):
    def __init__(self, tuple_of_modules):
        super().__init__()
        self.children = tuple_of_modules

        self.mass        = sum(child.mass        for child in self.children)
        self.sensitivity = sum(child.sensitivity for child in self.children)
        
    def forward(self, x, w):
        return tuple(child.forward(x, wi) for child, wi in zip(self.children, w))

    def initialize(self, device):
        return Vector(tuple(child.initialize(device) for child in self.children))

    def normalize(self, vector, target_norm=1):
        if self.mass > 0:
            normalized_child_vectors = []
            for vi, child in zip(vector, self.children):
                normalized_child_vectors.append(child.normalize(vi, target_norm=child.mass / self.mass * target_norm))
            return Vector(tuple(normalized_child_vectors))
        elif vector.length > 0:
            return vector * 0
        else:
            return vector


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.initialize = lambda device : Vector(None)
        self.normalize  = lambda vector, target_norm : Vector(None)
        self.alpha = alpha

    def forward(self, x, w):
        if isinstance(x, tuple):
            return tuple(self.forward(xi, w) for xi in x)
        else:
            return self.alpha * x


class Add(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.initialize = lambda device : Vector(None)
        self.normalize  = lambda vector, target_norm : Vector(None)

    def forward(self, x, w):
        assert isinstance(x, tuple), "can only compose add with tuples"
        return sum(xi for xi in x)
