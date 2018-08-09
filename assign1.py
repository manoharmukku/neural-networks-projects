import pandas as pd
import numpy as np
import sys
from sys import stdout
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Signum function
def signum(x):
    if (x > 0):
        return +1
    else:
        return -1

# Sanity check of command line arguments
if (len(sys.argv) != 5):
    print ("Usage: python assign1.py n_features n_samples learning_rate max_iterations")
    sys.exit()

# Get the parameters from command line
m = int(sys.argv[1]) # Number of features
n = int(sys.argv[2]) # Number of samples from command line
learning_rate = float(sys.argv[3]) # Learning rate from command line
max_iterations = int(sys.argv[4]) # Maximum number of iterations from command line

# Randomly generate two dimensional data using Normal distribution for two linearly separable classes
# Generate mean randomly for first class
mean1 = np.random.uniform(low = 5.0, high = 7.0, size = (m, ))
sd1 = 1.0

# Generate mean randomly for second class
mean2 = np.random.uniform(low = -2.0, high = 0.0, size = (m, ))
sd2 = 1.0

class1_input_data = np.random.normal(mean1, sd1, ((int)(0.5*n), m)) # (n/2, m) matrix having desired output label +1
class2_input_data = np.random.normal(mean2, sd2, ((int)(0.5*n), m)) # (n/2, m) matrix having desired output label -1

class1_desired_labels = np.full(((int)(0.5*n), 1), +1) # Class1 labels = +1
class2_desired_labels = np.full(((int)(0.5*n), 1), -1) # Class2 labels = -1

bias_input_term = np.full(((int)(0.5*n), 1), +1) # bias input always = +1

# Contatenate all into one
class1_data = np.concatenate((bias_input_term, class1_input_data, class1_desired_labels), axis = 1)
class2_data = np.concatenate((bias_input_term, class2_input_data, class2_desired_labels), axis = 1)

# Concatenate both the classes
data = np.concatenate((class1_data, class2_data), axis = 0)

# Randomly initialize free parameters between -0.05 and 0.05
weight_vector = np.random.uniform(low = -100.0, high = -99.0, size = (m+1, 1)) # (m+1, 1) weight vector including the bias term
weight_vector[0] = -1000.0

print ("============================")
print ("{}D data".format(m))
print ("============================\n")

print ("Number of featues: {}".format(m))
print ("Number of samples in class 1: {}".format((int)(0.5*n)))
print ("Number of samples in class 2: {}".format((int)(0.5*n)))
print ("Learning rate: {}".format(learning_rate))

print ("\nTraining on {}D data:".format(m))
print ("-----------------------")

fig = plt.figure()

for i in range(max_iterations):
    stdout.write ("\rIteration %d of %d (max)"%(i+1, max_iterations))
    stdout.flush()

    # Save the weights of this iteration
    weights_prev = weight_vector.copy()

    # Plot the decision boundary before this iteration (only for 2D data)
    if (m == 2):
        b = weight_vector[0][0]
        w1 = weight_vector[1][0]
        w2 = weight_vector[2][0]
        x = np.linspace(-3, 7)
        y = (-w1/w2) * x + (-b/w2)

        plt.scatter(class1_input_data[:,0], class1_input_data[:,1], c='blue', marker = '+')
        plt.scatter(class2_input_data[:,0], class2_input_data[:,1], c='green', marker='o')
        plt.plot(x, y, 'k-')
        plt.pause(0.00001)
        plt.close()

    # Randomly shuffle the data
    np.random.shuffle(data)

    # Iterate over all the rows of the data and train the perceptron model
    for row in data:
        input_vector = row[0:m+1].T.reshape(m+1, 1) # (m+1, 1) input vector including +1 bias term
        desired_output = row[m+1] # Desired output corresponding to the input vector, +1 or -1

        # Calculate actual output
        actual_output = signum((weight_vector.T).dot(input_vector))

        # Update the weights
        weight_vector += (learning_rate * (desired_output - actual_output) * input_vector)

    # If no change in weights from previous iteration
    if (np.array_equal(weight_vector.flatten(), weights_prev.flatten())):
        break

print ("\nDone training\n")

# Print the equation of the decision bounday
print ("The final weight matrix is: ")
print (weight_vector.T)

# Plot the decision boundary (if 2D data)
if (m == 2):
    b = weight_vector[0][0]
    w1 = weight_vector[1][0]
    w2 = weight_vector[2][0]
    x = np.linspace(-3, 7)
    y = (-w1/w2) * x + (-b/w2)

    plt.scatter(class1_input_data[:,0], class1_input_data[:,1], c='blue', marker = '+')
    plt.scatter(class2_input_data[:,0], class2_input_data[:,1], c='green', marker='o')
    plt.plot(x, y, 'k-')
    plt.show()