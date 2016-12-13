import utils
import numpy as np
import numpy.linalg as lg
import math as ma


def givence(dim, i, j, sin, cos):
	matrix = np.diag([1. for k in range(dim)])
	matrix[i, i], matrix[j, j] = cos, cos
	matrix[i, j], matrix[j, i] = -sin, sin
	return matrix


def main():
	with open('data/input.txt', mode='r') as file:
		m = np.matrix(utils.parse(file))
		f = np.matrix(utils.parse(file)).T
	a, b = np.matrix(np.copy(m)), np.matrix(np.copy(f))
	dim = len(m)
	for i in range(dim):
		for j in range(i + 1, dim):
			r = ma.sqrt(m[i, i] ** 2 + m[j, i] ** 2)
			sin, cos = -m[j, i] / r, m[i, i] / r
			t = givence(dim, i, j, sin, cos)
			m = np.dot(t, m)
			f = np.dot(t, f)
	x = np.array([float(0) for i in range(dim)]).T
	for k in reversed(range(dim)):
		s = sum([x[i] * m[k, i] for i in range(k, dim)])
		x[k] = (f[k] - s) / m[k, k]
	r = np.array(np.dot(a, x).T - b).T[0]
	with open('summary/rotations.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in x]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
	pass


if __name__ == '__main__':
	main()
