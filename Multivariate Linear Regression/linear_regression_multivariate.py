import numpy as np
import matplotlib.pyplot as plt 

def run():
	np.set_printoptions(precision=3)
	np.set_printoptions(suppress=True)

	data = np.mat("1.0,2.,10;100,200,1000;3000,6000,30000")

	normalized_data = scale_and_mean_normalize_features(data)
	print()
	print("data\n"+str(normalized_data))

	theta = np.mat("0;0;0")
	print("error initial: " + str(calculate_error_of_theta(theta, normalized_data)))


	learning_factor = 0.1
	new_theta = run_gradient_descent(theta,normalized_data, learning_factor, 100)

	print("\nnew theta\n"+str(new_theta))
	print("error new: " + str(calculate_error_of_theta(new_theta, normalized_data)))

# updates theta using gradient descent until convergence
def run_gradient_descent(theta, data, learning_factor, num_iterations):
	new_theta = theta

	last_error = 9999999999999
	new_error = 999999999999
	error_of_num_iterations = [[], []]
	i = 0

	while not convergence(last_error, new_error):
		new_theta = gradient_descent_step(new_theta,data, learning_factor)

		last_error = new_error
		new_error = calculate_error_of_theta(new_theta, data)

		error_of_num_iterations[0].append(i)
		error_of_num_iterations[1].append(new_error)

		i += 1


	plt.plot(error_of_num_iterations[0],error_of_num_iterations[1])
	plt.show()
	return new_theta

# returns theta after altering it once
def gradient_descent_step(theta,data, learning_factor):
	m = data.shape[0]
	n = theta.shape[0]

	new_theta = np.zeros(theta.shape)

	for j in range(n):
	
		derivative_of_theta_j = 0
		theta_j = np.asscalar(theta[np.ix_([j],[0])])

		for i in range(m):
			x = get_x_subscript_i_of_data(data, i)
			x_of_j = np.asscalar(x[np.ix_([j],[0])])
			y_real = np.asscalar(data[np.ix_([i],[n-1])])
			y_pred = get_y_of_hypothesis(theta, x)

			derivative_of_theta_j += (y_pred - y_real) * x_of_j

		new_theta[j][0] = theta_j - learning_factor * ((1/m) * derivative_of_theta_j)

	return new_theta

## returns J(theta)
def calculate_error_of_theta(theta, data):
	m = data.shape[0]
	n = data.shape[1]
	total_error = 0

	for i in range(m):
		x = get_x_subscript_i_of_data(data, i)
		y_pred = get_y_of_hypothesis(theta,x)
		y_real = np.asscalar(data[np.ix_([i],[n-1])])

		total_error += (y_pred - y_real)**2

	error = (1/(2*m)) * total_error

	return error

# returns a vector of all x values of one tuple
def get_x_subscript_i_of_data(data, i):
	n = (data.shape[1]) -1
	x_i = data[i:i+1, 0:n]

	x0 = np.ones((1,1))

	x_i = np.hstack((x0, x_i))

	return x_i.transpose()

# returns h(x)
def get_y_of_hypothesis(theta, x):
	y = theta.transpose() * x

	return np.asscalar(y)

def convergence(old_error, new_error):
	decrease = old_error - new_error

	if decrease < 0.001:
		return True

	return False

##
## Mean normalization and featuer scaling
##

# returns the mean of a feature
def find_mean_of_xj(data, j):
	m = data.shape[0]
	xs_j = data[0:m, j:j+1]

	sum = xs_j.sum()
	m = data.shape[0]
	mean = sum / m

	return mean

# returns the standard deviation of a feature
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
			normalized_data[i,j] = (x - mean) / standard_deviation

	return normalized_data


if __name__ == '__main__':
	run()