import torch

def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two Vectors."""
    v1_size = (v1**2).sum()**0.5
    v2_size = (v2**2).sum()**0.5
    if v1_size > 0 and v2_size > 0:
        return (v1 * v2).sum() / (v1_size * v2_size)
    else:
        return 0


class Vector:
    """For doing algebra on tensors and trees of tensors.

    An instance of Vector stores either None, a tensor, or a tuple of
    Vectors. Vectors can be added, subtracted and scalar-multiplied.
    We also support elementwise multiplication and division for
    convenience.

    Vectors are intended to store the weights of a neural net,
    allowing weight updates to be implemented using simple algebra.
    """

    def __init__(self, v):
        """Stores a tensor or tuple of Vectors."""
        self.weight = v

        if self.weight is None:
            self.length = 0
        elif isinstance(self.weight, tuple):
            self.length = sum(wi.length for wi in self.weight)
        else:
            self.length = 1

    def grad(self):
        """Returns the gradient Vector of this Vector."""
        if self.weight is None:
            return Vector(None)
        elif isinstance(self.weight, tuple):
            return Vector(tuple(wi.grad() for wi in self.weight))
        else:
            return Vector(self.weight.grad)

    def empty(self):
        """Returns an empty Vector with the same structure as this Vector."""
        if self.weight is None:
            return Vector(None)
        elif isinstance(self.weight, tuple):
            return Vector(tuple(wi.empty() for wi in self.weight))
        else:
            return Vector(torch.empty_like(self.weight))

    def zero_grad(self):
        """Sets all gradients of this Vector to None."""
        if self.weight is None:
            pass
        elif isinstance(self.weight, tuple):
            for wi in self.weight:
                wi.zero_grad()
        else:
            self.weight.grad = None

    def require_grad(self):
        """Sets all tensors of this Vector to require grad."""
        if self.weight is None:
            pass
        elif isinstance(self.weight, tuple):
            for wi in self.weight:
                wi.require_grad()
        else:
            self.weight.requires_grad = True

    def flatten(self):
        """Returns a list of all tensors in this Vector."""
        tensor_list = []
        if self.weight is None:
            pass
        elif isinstance(self.weight, tuple):
            for wi in self.weight:
                tensor_list += wi.flatten()
        else:
            tensor_list += [self.weight]
        assert len(tensor_list) == self.length
        return tensor_list

    def assign(self, tensor_list):
        """Sets the weights of this Vector from a list of tensors."""
        assert len(tensor_list) == self.length

        if self.weight is None:
            pass
        elif isinstance(self.weight, tuple):
            for wi in self.weight:
                wi.assign(tensor_list[:wi.length])
                tensor_list = tensor_list[wi.length:]
        else:
            self.weight = tensor_list[0]

        return self

    def sum(self):
        """Computes the sum of all entries in the Vector."""
        if self.weight is None:
            return 0
        elif isinstance(self.weight, tuple):
            return sum(wi.sum() for wi in self.weight)
        else:
            return self.weight.sum().item()

    def __getitem__(self, item):
        """Allows Vectors of tuples to be indexed."""
        assert isinstance(self.weight, tuple)
        return self.weight[item]

    def __str__(self):
        """Lets us print the Vector."""
        if isinstance(self.weight, tuple):
            string = "("
            for wi in self.weight:
                string += wi.__str__() +","
            string += ")"
            return string
        else:
            return str(self.weight)

    def __add__(self, other):
        """Add two Vectors with the same subtructure."""
        tensor_list = torch._foreach_add(self.flatten(), other.flatten())
        return self.empty().assign(tensor_list)

    def __mul__(self, other):
        """Multiply a Vector by a scalar or elementwise multiply two Vectors."""
        tensor_list = torch._foreach_mul(self.flatten(), other.flatten() if isinstance(other, Vector) else other)
        return self.empty().assign(tensor_list)

    def __sub__(self, other):
        """Subtract a Vector from another Vector."""
        tensor_list = torch._foreach_sub(self.flatten(), other.flatten())
        return self.empty().assign(tensor_list)

    def __rmul__(self, other):
        """Multiply by a scalar on the left."""
        return self * other

    def __truediv__(self, other):
        """Scalar or elementwise division."""
        tensor_list = torch._foreach_div(self.flatten(), other.flatten() if isinstance(other, Vector) else other)
        return self.empty().assign(tensor_list)

    def __rtruediv__(self, other):
        """Allow reciprocals."""
        return other * self ** -1.0

    def __pow__(self, other):
        """Elementwise powers."""
        tensor_list = torch._foreach_pow(self.flatten(), other)
        return self.empty().assign(tensor_list)


if __name__ == "__main__":
    t = torch.tensor(2.0)

    a = Vector(t)
    b = Vector(None)
    a = Vector((a,b))
    a = Vector((a,a))

    print(a)                       # 2
    print(a + (a*2)**3 - 3*a)      # 60
    print(1/a)                     # 1/2
    print(a/a)                     # 1
    print(a-a)                     # 0
    print(a*a*a-a)                 # 6
    print(a/a**0.5)                # sqrt(2)
        
    t = torch.tensor(2.0, requires_grad=True)
    a = Vector(t); a = Vector((a,a)); a = Vector((a,a))
    print(a)

    with torch.no_grad():
        a -= a/2
        a.require_grad()

        print(a)                   # 1, requires_grad=True
