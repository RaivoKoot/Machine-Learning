import numpy as np

#              j0    j1    j2
data = np.mat("1    ,2    ,10;"
			  "100  ,200  ,1000;"
			  "200  ,400  ,2000;"
			  "3000 ,6000 ,30000")

print("*******************")
print(data)

dimensions = data.shape

rows = dimensions[0]
columns = dimensions[1] 

# get feature x at row i and column j
i = 3
j = 2
x_i_j = data[np.ix_([i],[j])]
x_i_j_alt = data[3,0]

# get all features at row i
i = 2
x_i = data[i:i+1, 0:columns].transpose()

# get all instances of feature x
j = 1
x_j = data[0:rows, j:j+1]


print("******************************")
print("one feature x: "+str(x_i_j))
print("alt: "+str(x_i_j_alt))
print()

print("x subscript i: "+str(x_i))
print()

print("all x of j: "+str(x_j))

