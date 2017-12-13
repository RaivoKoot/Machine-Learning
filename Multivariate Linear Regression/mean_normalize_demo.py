import numpy as np

def find_mean_of_xj(data, j):
	m = data.shape[0]
	xs_j = data[0:m, j:j+1]

	sum = xs_j.sum()
	m = data.shape[0]
	mean = sum / m

	return mean

def calc_standard_deviation(data, j):
	m = data.shape[0]
	mean = find_mean_of_xj(data,j)

	xs_j = data[0:m, j:j+1]

	standard_deviation = 0
	for i in range(m):
		standard_deviation += (np.asscalar(xs_j[i][0]) - mean) ** 2

	standard_deviation /= m
	standard_deviation **=(1/2)

	return standard_deviation

def scale_and_mean_normalize_features(data):
	n = data.shape[1] - 1
	m = data.shape[0]

	normalized_data = data.copy()

	for j in range(n):
		mean = find_mean_of_xj(data, j)
		standard_deviation = calc_standard_deviation(data, j)

		for i in range(m):
			x = data[i,j]
			print("Element at i "+str(i)+" and j "+str(j)+" is "+str(x))
			normalized_data[i,j] = (x - mean) / standard_deviation

	return normalized_data



def run():
	np.set_printoptions(precision=3)
	np.set_printoptions(suppress=True)

	data = np.mat("1.0,2,10;	100,200,1000;	3000,6000,30000")
	print(data)
	print()


	normalized_data = scale_and_mean_normalize_features(data)

	print()
	print(normalized_data)

	#data = np.mat("9; 2; 5; 4; 12; 7; 8; 11; 9; 3; 7; 4; 12; 5; 4; 10; 9; 6; 9; 4")
	print()
	print(find_mean_of_xj(data, 0))
	print(find_mean_of_xj(data, 1))
	print(find_mean_of_xj(data, 2))
	print()
	print(calc_standard_deviation(data, 0))
	print(calc_standard_deviation(data, 1))



if __name__ == '__main__':
	run()