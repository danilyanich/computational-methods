import utils
import numpy as np
import numpy.linalg as lg


class Help:
	def __init__(self, matrix, aug, dim):
		self.matrix = matrix
		self.aug = aug
		self.dim = dim
		pass

	def b(self, ind):
		if 0 <= ind < self.dim - 1:
			return -self.matrix[ind][ind + 1]
		return 0

	def c(self, ind):
		if 0 <= ind < self.dim:
			return self.matrix[ind][ind]
		return 0

	def a(self, ind):
		if 0 < ind < self.dim:
			return -self.matrix[ind][ind - 1]
		return 0

	def f(self, ind):
		if 0 <= ind < self.dim:
			return self.aug[ind]
		return 0

	pass


def check(array, index):
	if 0 <= index < len(array):
		return array[index]
	return 0


def main():
	m, f = [], []
	with open('data/3diag.txt', mode='r') as file:
		m = utils.parse(file)
		f = utils.parse(file)
	dim = len(m)
	h = Help(m, f[0], dim)
	al, be = [0 for i in range(dim)], [0 for i in range(dim)]
	for i in range(dim):
		de = h.c(i) - h.a(i) * check(al, i - 1)
		al[i] = h.b(i) / de
		be[i] = (h.f(i) + check(be, i - 1) * h.a(i)) / de
	x = [0 for i in range(dim)]
	for i in reversed(range(dim)):
		x[i] = al[i] * check(x, i + 1) + be[i]
	r = (np.dot(m, x).T - f)[0]
	with open('summary/tridiagonal.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5f}'.format(e)) for e in x]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
	pass


if __name__ == '__main__':
	main()
