import copy


class Module:
    def __init__(self):
        self.mass = None
        self.sensitivity = None
        self.length = None
        self.children = []
        
    def forward(self, x, w):
        raise NotImplementedError

    def initialize(self, device):
        raise NotImplementedError

    def normalize(self, w, target_norm):
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
        self.length = m0.length + m1.length
        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity
        
    def forward(self, x, w):
        m0, m1 = self.children
        w0 = w[:m0.length]
        w1 = w[m0.length:]
        return m1.forward(m0.forward(x, w0), w1)

    def initialize(self, device):
        m0, m1 = self.children
        return m0.initialize(device) + m1.initialize(device)

    def normalize(self, w, target_norm=1):
        m0, m1 = self.children
        w0 = w[:m0.length]
        w1 = w[m0.length:]

        if self.mass > 0:
            w0 = m0.normalize(w0, target_norm=m0.mass / self.mass * target_norm / m1.sensitivity)
            w1 = m1.normalize(w1, target_norm=m1.mass / self.mass * target_norm)
            return w0 + w1
        else:
            return [0] * self.length


class TupleModule(Module):
    def __init__(self, tuple_of_modules):
        super().__init__()
        self.children = tuple_of_modules
        self.length      = sum(child.length      for child in self.children)
        self.mass        = sum(child.mass        for child in self.children)
        self.sensitivity = sum(child.sensitivity for child in self.children)
        
    def forward(self, x, w):
        output = []
        for child in self.children:
            w_child = w[:child.length]
            output.append(child.forward(x, w_child))
            w = w[child.length:]
        return output

    def initialize(self, device):
        tensor_list = []
        for child in self.children:
            tensor_list += child.initialize(device)
        return tensor_list

    def normalize(self, w, target_norm=1):
        if self.mass > 0:
            tensor_list = []
            for child in self.children:
                w_child = w[:child.length]
                tensor_list += child.normalize(w_child, target_norm=child.mass / self.mass * target_norm)
                w = w[child.length:]
            return tensor_list
        else:
            return [0] * self.length


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.initialize = lambda device : []
        self.normalize  = lambda w, target_norm : []
        self.alpha = alpha

    def forward(self, x, _):
        if isinstance(x, list):
            return [self.forward(xi, _) for xi in x]
        else:
            return self.alpha * x


class Add(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.initialize = lambda device : []
        self.normalize  = lambda w, target_norm : []

    def forward(self, x, w):
        assert isinstance(x, list), "can only compose add with tuple modules"
        return sum(xi for xi in x)
