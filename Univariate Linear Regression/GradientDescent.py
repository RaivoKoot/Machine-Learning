from numpy import *
import matplotlib.pyplot as plt 


def gradient_descent_step(current_m, current_b,data, learning_factor):
	n = len(data)
	derivative_of_b = 0
	derivative_of_m = 0

	for i in range(0,len(data)):
		x = data[i][0]
		y = data[i][1]

		derivative_of_b += x * current_m + current_b - y
		derivative_of_m += (x * current_m + current_b - y) * x

	new_b = current_b - (learning_factor * ((1 / n) * derivative_of_b))
	new_m = current_m - (learning_factor * ((1 / n) * derivative_of_m))

	return [new_m, new_b]

def run_gradient_descent(current_m, current_b, data, learning_factor):
	parameters = [current_m,current_b]

	last_error = 9999999999999
	new_error = 999999999999
	xy = [[], []]
	i = 0

	while(not convergence(last_error,new_error)):
		new_m = parameters[0]
		new_b = parameters[1]
		parameters = gradient_descent_step(new_m, new_b, data, learning_factor)

		last_error = new_error
		new_error = calculate_error_for_regression_line(parameters[0], parameters[1], data)

		xy[0].append(i)
		xy[1].append(new_error)

		i += 1

	print("error: "+str(calculate_error_for_regression_line(parameters[0], parameters[1], data)))

	print("theta: "+str(parameters))
	plt.plot(xy[0],xy[1])

	plt.show()

	return parameters

def calculate_error_for_regression_line(m, b, data):
		n = len(data)

		total_error = 0

		for i in range(n):
			x = data[i][0]
			yReal = data[i][1]

			yPrediction = x * m + b

			rSquared = (yPrediction - yReal)** 2

			total_error += rSquared;
		

		error_mean = total_error / (n * 2);

		return error_mean;
def convergence(old_error, new_error):
	decrease = old_error - new_error

	print("decrease: "+str(decrease))
	if decrease < 0.0001:
		return True

	return False

def run():

	data = [[1,2],[2,4]]
	data = genfromtxt("data.csv", delimiter=",")


	m = 0
	b = 0

	print(calculate_error_for_regression_line(m, b, data))
	run_gradient_descent(m,b,data,0.0000000001)


if __name__ == '__main__':
    run()