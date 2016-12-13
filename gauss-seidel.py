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
	m_t = m.T
	m = np.dot(m_t, m)
	f = np.dot(m_t, f)
	iteration = 0
	while lg.norm(x - x_1, ord=np.inf) >= eps:
		x_1 = np.copy(x)
		for i in range(len(x)):
			x[i] -= float(sum([x[j] * m[i, j] for j in range(len(x))]) - f[i]) / m[i, i]
		iteration += 1
	r = np.array(np.dot(a, x) - b).T[0]
	with open('summary/gauss-seidel.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in np.array(x).T[0]]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
		file.write('\niteration: {:d}'.format(iteration))
	pass


if __name__ == '__main__':
	main()
