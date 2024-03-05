import copy
import torch


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mass = None
        self.sensitivity = None
        self.weight = None
        
    def forward(self, x):
        raise NotImplementedError
    
    def norm(self, w):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def update(self, lr, beta, wd):
        raise NotImplementedError

    def __str__(self):
        return f"Module of mass {self.mass} and sensitivity {self.sensitivity}."

    def __matmul__(self, other):
        return CompositeModule(self, other)

    def __add__(self, other):
        return SumModule(self, other)

    def __rmul__(self, other):
        assert other != 0
        return ScalarMultiply(other) @ self

    def __pow__(self, other):
        assert other >= 0 and other % 1 == 0
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
    
    def norm(self, w):
        if self.m0.mass == 0:
            return self.m1.norm(w[1])
        elif self.m1.mass == 0:
            return self.m0.norm(w[0]) * self.m1.sensitivity
        else:
            c0 = self.mass / self.m0.mass * self.m1.sensitivity
            c1 = self.mass / self.m1.mass
            return max(c0 * self.m0.norm(w[0]), c1 * self.m1.norm(w[1]))

    def initialize(self):
        self.m0.initialize()
        self.m1.initialize()
        self.weight = torch.nn.ParameterList((self.m0.weight, self.m1.weight))

    def update(self, lr, beta, wd):
        if self.m0.mass == 0:
            self.m1.update(lr, beta, wd)
        elif self.m1.mass == 0:
            self.m0.update(lr / self.m1.sensitivity, beta, wd)
        else:
            c0 = self.mass / self.m0.mass * self.m1.sensitivity
            c1 = self.mass / self.m1.mass
            self.m0.update(lr / c0, beta, wd)
            self.m1.update(lr / c1, beta, wd)


class SumModule(Module):
    def __init__(self, m0, m1):
        super().__init__()
        self.m0 = m0
        self.m1 = m1

        self.mass = m0.mass + m1.mass
        self.sensitivity = m0.sensitivity + m1.sensitivity
        
    def forward(self, x):
        return self.m0.forward(x) + self.m1.forward(x)
    
    def norm(self, w):
        if self.m0.mass == 0:
            return self.m1.norm(w[1])
        elif self.m1.mass == 0:
            return self.m0.norm(w[0])
        else:
            c0 = self.mass / self.m0.mass
            c1 = self.mass / self.m1.mass
            return max(c0 * self.m0.norm(w[0]), c1 * self.m1.norm(w[1]))

    def initialize(self):
        self.m0.initialize()
        self.m1.initialize()
        self.weight = torch.nn.ParameterList((self.m0.weight, self.m1.weight))

    def update(self, lr, beta, wd):
        if self.m0.mass == 0:
            self.m1.update(lr, beta, wd)
        elif self.m1.mass == 0:
            self.m0.update(lr, beta, wd)
        else:
            c0 = self.mass / self.m0.mass
            c1 = self.mass / self.m1.mass
            self.m0.update(lr / c0, beta, wd)
            self.m1.update(lr / c1, beta, wd)


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.forward = lambda x: alpha * x
        self.initialize = lambda : None
