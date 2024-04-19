import torch


class Vector:
    """For doing algebra on lists of tensors.

    An instance of Vector stores a list of tensors. Vectors can be 
    added, subtracted, scalar-multiplied, elementwise-multiplied, etc.
    We also support in-place operations for efficiency.

    Vectors are intended to store the weights of a neural net,
    allowing weight updates to be implemented using simple algebra.
    """

    def __init__(self, tensor_list):
        """Stores a list of tensors."""
        self.tensor_list = tensor_list

    def __getitem__(self, item):
        """Allows Vectors to be indexed and looped over."""
        return self.tensor_list[item]

    def grad(self):
        """Returns the gradient list of this Vector."""
        return Vector([tensor.grad for tensor in self])

    def zero_grad(self):
        """Delete the gradients of this Vector."""
        for tensor in self:
            tensor.grad = None

    def __str__(self):
        """Lets us print the Vector."""
        return str([t for t in self])

    def __iadd__(self, other):
        """In-place add."""
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_add_(self.tensor_list, other)
        return self

    def __add__(self, other):
        """Add."""
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_add(self.tensor_list, other)
        return Vector(new_list)

    def __mul__(self, other):
        """Multiply."""
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_mul(self.tensor_list, other)
        return Vector(new_list)

    def __rmul__(self, other):
        """Multiply from the left."""
        return self * other

    def __imul__(self, other):
        """In-place multiply."""
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_mul_(self.tensor_list, other)
        return self

    def __isub__(self, other):
        """In-place subtract."""
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_sub_(self.tensor_list, other)
        return self

    def __sub__(self, other):
        """Subtract."""
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_sub(self.tensor_list, other)
        return Vector(new_list)

    def __itruediv__(self, other):
        """In-place division."""
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_div_(self.tensor_list, other)
        return self

    def __truediv__(self, other):
        """Division."""
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_div(self.tensor_list, other)
        return Vector(new_list)

    def __ipow__(self, other):
        """In-place power."""
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_pow_(self.tensor_list, other)
        return self

    def __pow__(self, other):
        """Power."""
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_pow(self.tensor_list, other)
        return Vector(new_list)


if __name__ == "__main__":
    
    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a *= 2;  print(a)
    a += 1;  print(a)
    a -= 1;  print(a)
    a /= 2;  print(a)
    a **= 2; print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a **= a; print(a)
    a *= a;  print(a)   
    a /= a;  print(a)
    a += a;  print(a)
    a -= a;  print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a = a * 2;  print(a)
    a = a + 1;  print(a)
    a = a - 1;  print(a)
    a = a / 2;  print(a)
    a = a ** 2; print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a = a * a;  print(a)
    a = a + a;  print(a)
    a = a / a;  print(a)
    a = a ** a; print(a)
    a = a - a;  print(a)