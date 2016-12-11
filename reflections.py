import utils
import numpy as np
import numpy.linalg as lg
import math as ma


def omega(s, e):
	al = ma.sqrt(float(np.dot(s.T, s)))
	kappa = 1 / ma.sqrt(float(np.dot(s.T, s - al * e)) * 2)
	return np.matrix((s - al * e) * kappa)


def main():
	m, f = [float], [float]
	with open('data/input.txt', mode='r') as file:
		m = np.matrix(utils.parse(file))
		f = np.matrix(utils.parse(file)).T
	a, b = np.matrix(np.copy(m)), np.matrix(np.copy(f))
	dim = len(m)
	for k in range(dim - 1):
		s = np.matrix([m[i, k] if i >= k else 0 for i in range(dim)]).T
		e = np.matrix([0 if i != k else 1 for i in range(dim)]).T
		try:
			w = omega(s, e)
			V = np.matrix(np.diag([float(1) for i in range(dim)]) - 2 * np.dot(w, w.T))
			m -= 2 * np.dot(w, np.dot(w.T, m))
			f -= 2 * np.dot(w, np.dot(w.T, f))
		except ZeroDivisionError:
			print(ZeroDivisionError)
	x = np.array([float(0) for i in range(dim)]).T
	for k in reversed(range(dim)):
		s = sum([x[i] * m[k, i] for i in range(k, dim)])
		x[k] = (f[k] - s) / m[k, k]
	r = np.array(np.dot(a, x).T - b).T[0]
	with open('summary/reflections.txt', mode='w+') as file:
		file.write('x:')
		[file.write('{:<15.5e}'.format(e)) for e in x]
		file.write('\nr:')
		[file.write('{:<15.5e}'.format(e)) for e in r]
		file.write('\n||r|| = {:e}'.format(lg.norm(r.T, ord=np.inf)))
	pass


if __name__ == '__main__':
	main()
