import utils
import numpy as np
import numpy.linalg as lg


def main():
	m, f = [], []
	with open('data/input.txt', mode='r') as file:
		m = np.matrix(utils.parse(file))
		f = np.matrix(utils.parse(file)).T
	a, b, eps = np.matrix(np.copy(m)), np.matrix(np.copy(f)), 1.0e-5
	x, x_1 = np.array(f), np.array([0. for i in range(len(m))])
	for i in range(len(m)):
		f[i] /= m[i, i]
		m[i] /= -m[i, i]
		m[i, i] = 0
	iteration = 0
	while lg.norm(x - x_1, ord=np.inf) >= eps:
		x_1 = x
		x = np.dot(m, x) + f
		iteration += 1
		pass
	r = np.array(np.dot(a, x) - b).T[0]
	with open('summary/jacobi.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in np.array(x).T[0]]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
		file.write('\niteration: {:d}'.format(iteration))


if __name__ == '__main__':
	main()
