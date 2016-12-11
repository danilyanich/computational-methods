import utils
import numpy as np
import numpy.linalg as lg
import math as ma


def main():
	m, f = [], []
	with open('data/input.txt', mode='r') as file:
		m = np.matrix(utils.parse(file))
		f = np.matrix(utils.parse(file)).T
	a, b = np.matrix(np.copy(m)), np.matrix(np.copy(f))
	eps, norm = 1.0e-5, lg.norm(m, ord=np.inf) * lg.norm(f, ord=np.inf)
	x, x_1 = np.array(f), np.array([0. for i in range(len(m))])
	m /= norm
	f /= norm
	m = np.diag([1 for i in range(len(m))]) - m
	iteration = 0
	while lg.norm(x - x_1, ord=1) >= eps:
		x_1 = x
		x = np.dot(m, x) + f
		iteration += 1
		pass
	r = np.array(np.dot(a, x) - b).T[0]
	with open('summary/iterations.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in np.array(x)[0]]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
		file.write('\niteration: {:d}'.format(iteration))
		file.write(
			'\nk: {:.0f}'.format((ma.log(eps) + ma.log(1 - lg.norm(m, ord=np.inf)) - ma.log(lg.norm(f, ord=np.inf))) / (
				(ma.log(lg.norm(m, ord=np.inf)) - 1))))


if __name__ == '__main__':
	main()
