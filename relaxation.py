import utils
import numpy as np
import numpy.linalg as lg


def main():
	m, f = [], []
	with open('data/input.txt', mode='r') as file:
		m = np.matrix(utils.parse(file))
		f = np.matrix(utils.parse(file)).T
	a, b = np.matrix(np.copy(m)), np.matrix(np.copy(f))
	x, x_1, eps = np.array(f), np.array([0. for i in range(len(m))]), 1.0e-5
	m = np.matrix(np.dot(m.T, m))
	f = np.matrix(np.dot(m.T, f))
	m = np.matrix(np.diag([1 for i in range(len(m))]) - m)
	iteration, omega = 0, 1
	while lg.norm(x - x_1, ord=np.inf) / omega >= eps:
		x_1 = np.copy(x)
		for i in range(len(x)):
			x[i] = (1 - omega) * x[i] - omega * (sum([m[i, j] * x[j] for j in range(len(x))]) + f[i]) / m[i, i]
		iteration += 1
		print(x)
		pass
	r = np.array(np.dot(a, x) - b).T[0]
	with open('summary/relaxation.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in np.array(x.T)[0]]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
		file.write('\niteration: {:d}'.format(iteration))
	pass


if __name__ == '__main__':
	main()
