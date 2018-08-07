import pandas as pd
import numpy as np
from sys import stdout

# Signum function
def signum(x):
    if (x > 0):
        return +1
    else:
        return -1

#### For 2D (2 featurs) data ####

# Parameters
m = 2 # Number of features
n = 10000 # Number of samples
learning_rate = 0.1
max_iterations = 10000 # Maximum number of iterations

# Randomly generate two dimensional data using Normal distribution for two linearly separable classes
mean1, sd1 = (8.0, 10.0), 1.0 # Mean and standard deviation for class 1 2d features
mean2, sd2 = (-7.0, -6.0), 1.0 # Mean and standard deviation for class 2 2d features
class1_input_data = np.random.normal(mean1, sd1, ((int)(0.5*n), m)) # (5000, 2) matrix having desired output label +1
class2_input_data = np.random.normal(mean2, sd2, ((int)(0.5*n), m)) # (5000, 2) matrix having desired output label -1

class1_desired_labels = np.full(((int)(0.5*n), 1), +1) # Class1 labels = +1
class2_desired_labels = np.full(((int)(0.5*n), 1), -1) # Class2 labels = -1

bias_input_term = np.full(((int)(0.5*n), 1), +1) # bias input always = +1

# Contatenate all into a single numpy nd-array
class1_data = np.concatenate((bias_input_term, class1_input_data, class1_desired_labels), axis = 1)
class2_data = np.concatenate((bias_input_term, class2_input_data, class2_desired_labels), axis = 1)
data = np.concatenate((class1_data, class2_data), axis = 0)

# Randomly initialize free parameters between -0.05 and 0.05
weight_vector = np.random.uniform(low = -0.05, high = 0.05, size = (m+1, 1)) # (m+1, 1) weight vector including the bias term

print ("============================")
print ("2D data")
print ("============================\n")

print ("Number of featues: {}".format(m))
print ("Number of samples in class 1: {}".format((int)(0.5*n)))
print ("Number of samples in class 2: {}".format((int)(0.5*n)))
print ("Learning rate: {}".format(learning_rate))

print ("\nTraining on 2D data...\n")
# print ("--------")

# for i in range(max_iterations):
# stdout.write ("\rIteration %d of %d (max)"%(i+1, max_iterations))
# stdout.flush()

# Save the weights of this iteration
# weights_prev = weight_vector

# Randomly shuffle the data
np.random.shuffle(data)

# Iterate over all the rows of the data and train the perceptron model
for row in data:
    input_vector = row[0:m+1].T.reshape(m+1, 1) # (m+1, 1) input vector including +1 bias term
    desired_output = row[m+1] # Desired output corresponding to the input vector, +1 or -1

    # Calculate actual output
    actual_output = signum((weight_vector.T).dot(input_vector))

    # Update the weights
    weight_vector += learning_rate * (desired_output - actual_output) * input_vector

# If no change in weights from previous iteration
# if (np.array_equal(weight_vector.reshape(m+1), weights_prev.reshape(m+1))):
#     break

print ("Done training\n")

# Print the equation of the decision bounday
print ("Equation of decision boundary:")
print ("------------------------------\n")
b = weight_vector[0][0]
w1 = weight_vector[1][0]
w2 = weight_vector[2][0]
print ("({}) * x1 + ({}) * x2 + ({}) = 0\n".format(w1, w2, b))


#### For 3D (3 features) data ####

# Parameters
m = 3 # Number of features
n = 10000 # Number of samples
learning_rate = 0.1
max_iterations = 10000 # Maximum number of iterations

# Randomly generate two dimensional data using Normal distribution for two linearly separable classes
mean1, sd1 = (8.0, 10.0, 9.0), 1.0 # Mean and standard deviation for class 1 3d features
mean2, sd2 = (-7.0, -6.0, -6.5), 1.0 # Mean and standard deviation for class 2 3d features
class1_input_data = np.random.normal(mean1, sd1, ((int)(0.5*n), m)) # (5000, 3) matrix having desired output label +1
class2_input_data = np.random.normal(mean2, sd2, ((int)(0.5*n), m)) # (5000, 3) matrix having desired output label -1

class1_desired_labels = np.full(((int)(0.5*n), 1), +1) # Class1 labels = +1
class2_desired_labels = np.full(((int)(0.5*n), 1), -1) # Class2 labels = -1

bias_input_term = np.full(((int)(0.5*n), 1), +1) # bias input always = +1

# Contatenate all into a single numpy nd-array
class1_data = np.concatenate((bias_input_term, class1_input_data, class1_desired_labels), axis = 1)
class2_data = np.concatenate((bias_input_term, class2_input_data, class2_desired_labels), axis = 1)
data = np.concatenate((class1_data, class2_data), axis = 0)

# Randomly initialize free parameters between -0.05 and 0.05
weight_vector = np.random.uniform(low = -0.05, high = 0.05, size = (m+1, 1)) # (m+1, 1) weight vector including the bias term

print ("============================")
print ("3D data")
print ("============================\n")

print ("Number of featues: {}".format(m))
print ("Number of samples in class 1: {}".format((int)(0.5*n)))
print ("Number of samples in class 2: {}".format((int)(0.5*n)))
print ("Learning rate: {}".format(learning_rate))

print ("\nTraining on 3D data...\n")

# print ("--------")

# for i in range(max_iterations):
# stdout.write ("\rIteration %d of %d (max)"%(i+1, max_iterations))
# stdout.flush()

# Save the weights of this iteration
# weights_prev = weight_vector

# Randomly shuffle the data
np.random.shuffle(data)

# Iterate over all the rows of the data and train the perceptron model
for row in data:
    input_vector = row[0:m+1].T.reshape(m+1, 1) # (m+1, 1) input vector including +1 bias term
    desired_output = row[m+1] # Desired output corresponding to the input vector, +1 or -1

    # Calculate actual output
    actual_output = signum((weight_vector.T).dot(input_vector))

    # Update the weights
    weight_vector += learning_rate * (desired_output - actual_output) * input_vector

# If no change in weights from previous iteration
# if (np.array_equal(weight_vector.reshape(m+1), weights_prev.reshape(m+1))):
#     break

print ("Done training\n")

# Print the equation of the decision bounday
print ("Equation of decision boundary:")
print ("------------------------------\n")
b = weight_vector[0][0]
w1 = weight_vector[1][0]
w2 = weight_vector[2][0]
w3 = weight_vector[3][0]
print ("({}) * x1 + ({}) * x2 + ({}) * x3 + ({}) = 0\n".format(w1, w2, w3, b))

