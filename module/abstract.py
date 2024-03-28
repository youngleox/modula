import copy


class Module:
    def __init__(self):
        super().__init__()
        self.mass = None
        self.sensitivity = None
        self.parameters = []
        self.m0, self.m1 = None, None
        
    def forward(self, x):
        raise NotImplementedError

    def initialize(self, device):
        pass

    def update(self, lr, hps):
        pass

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            if self.m0 is not None: self.m0.tare(relative = relative)
            if self.m1 is not None: self.m1.tare(relative = relative)
        else:
            self.tare(relative = absolute / self.mass)

    def print_submodules(self):
        if self.m0 is not None: self.m0.print_submodules()
        if self.m1 is not None: self.m1.print_submodules()

    def __str__(self):
        return f"Module of mass {self.mass} and sensitivity {self.sensitivity}."

    def __call__(self, x):
        return self.forward(x)

    def __matmul__(self, other):
        return CompositeModule(self, other)

    def __add__(self, other):
        return Add() @ TupleModule(self, other)

    def __rmul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return ScalarMultiply(other) @ self

    def __pow__(self, other):
        assert other >= 0 and other % 1 == 0, "nonnegative integer powers only"
        if other > 0:
            return self @ copy.deepcopy(self) ** (other - 1)
        else:
            return ScalarMultiply(1.0)


class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.m0 = m0
        self.m1 = m1

        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity
        
    def forward(self, x):
        return self.m1.forward(self.m0.forward(x))

    def initialize(self, device):
        self.m0.initialize(device)
        self.m1.initialize(device)
        self.parameters = self.m0.parameters + self.m1.parameters

    def update(self, lr, hps):
        if self.mass > 0:
            self.m0.update(self.m0.mass / self.mass / self.m1.sensitivity * lr, hps)
            self.m1.update(self.m1.mass / self.mass                       * lr, hps)


class TupleModule(Module):
    def __init__(self, m0, m1):
        super().__init__()
        self.m0 = m0
        self.m1 = m1

        self.mass = m0.mass + m1.mass
        self.sensitivity = m0.sensitivity + m1.sensitivity
        
    def forward(self, x):
        return (self.m0.forward(x), self.m1.forward(x))

    def initialize(self, device):
        self.m0.initialize(device)
        self.m1.initialize(device)
        self.parameters = self.m0.parameters + self.m1.parameters

    def update(self, lr, hps):
        if self.mass > 0:
            self.m0.update(self.m0.mass / self.mass * lr, hps)
            self.m1.update(self.m1.mass / self.mass * lr, hps)


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.forward = lambda x: alpha * x


class Add(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.forward = lambda x: x[0] + x[1]
