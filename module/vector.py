class Vector:

	def __init__(self, v=None):

		if isinstance(v, tuple):
			self.weight = tuple(Vector(vi) for vi in v)
		else:
			self.weight = v

	def grad(self):
		new_vector = Vector()
		if isinstance(self.weight, tuple):
			new_vector.weight = tuple(wi.grad() for wi in self.weight)
		else:
			new_vector.weight = self.weight.grad
		return new_vector

	def __getitem__(self, item):
		assert isinstance(self.weight, tuple)
		return self.weight[item]

	def __str__(self):
		if isinstance(self.weight, tuple):
			string = "("
			for wi in self.weight:
				string += wi.__str__() +","
			string += ")"
			return string
		else:
			return str(self.weight)

	def __add__(self, other):
		new_vector = Vector()
		if isinstance(self.weight, tuple):
			new_vector.weight = tuple(wi + oi for wi, oi in zip(self.weight, other.weight))
		else:
			new_vector.weight = self.weight + other.weight
		return new_vector

	def __mul__(self, other):
		new_vector = Vector()
		if isinstance(self.weight, tuple):
			new_vector.weight = tuple(wi * other for wi in self.weight)
		else:
			new_vector.weight = self.weight * other
		return new_vector

	def __sub__(self, other):
		return self + (-1) * other

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		return self * (1/other)

	def __pow__(self, other):
		new_vector = Vector()
		if isinstance(self.weight, tuple):
			new_vector.weight = tuple(wi ** other for wi in self.weight)
		else:
			new_vector.weight = self.weight ** other
		return new_vector


if __name__ == "__main__":
	import torch
	for t in [1, torch.tensor(1), torch.tensor(1.0, requires_grad=True)]:
		print('\n',type(t))

		a = (t,(t,t,(t,)),t)
		va = Vector(a)
		print(va)

		vb = va + (va*2)**3 - 3*va
		print(vb)

		if isinstance(t, torch.Tensor) and t.requires_grad:
			print(va.grad())

			loss = t.square()
			loss.backward()

			print(va.grad())