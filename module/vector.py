class Vector:

	def __init__(self, v):

		self.weight = v

	def grad(self):
		if self.weight is None:
			return Vector(None)
		elif isinstance(self.weight, tuple):
			return Vector(tuple(wi.grad() for wi in self.weight))
		else:
			return Vector(self.weight.grad)

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
		if self.weight is None:
			return Vector(None)
		elif isinstance(self.weight, tuple):
			return Vector(tuple(wi + oi for wi, oi in zip(self.weight, other.weight)))
		else:
			return Vector(self.weight + other.weight)

	def __mul__(self, other):
		if self.weight is None:
			return Vector(None)
		elif isinstance(self.weight, tuple):
			return Vector(tuple(wi * other for wi in self.weight))
		else:
			return Vector(self.weight * other)

	def __sub__(self, other):
		return self + (-1) * other

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		return self * (1/other)

	def __pow__(self, other):
		if self.weight is None:
			return Vector(None)
		elif isinstance(self.weight, tuple):
			return Vector(tuple(wi ** other for wi in self.weight))
		else:
			return Vector(self.weight ** other)


if __name__ == "__main__":
	import torch
	for t in [1, torch.tensor(1), torch.tensor(1.0, requires_grad=True)]:
		print('\n',type(t))

		a = Vector(t)
		b = Vector(None)
		a = Vector((a,b))
		a = Vector((a,a))


		print(a)

		b = a + (a*2)**3 - 3*a
		
		print(b)

		if isinstance(t, torch.Tensor) and t.requires_grad:
			print(a.grad())

			loss = t.square()
			loss.backward()

			print(a.grad())