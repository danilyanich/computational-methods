import numpy


def parse(file):
	dim = int(file.readline())
	matrix = []
	for i in range(dim):
		matrix.append([float(x) for x in file.readline().split()])
	return matrix
