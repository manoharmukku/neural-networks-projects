import pandas as pd
import numpy as np

# Randomly generate two dimensional data using Normal distribution for two linearly separable classes
n = 10000 # Number of samples
m = 2 # Number of features
mean1, sd1 = (8.0, 10.0), 1.0 # Mean and standard deviation for class 1 2d features
mean2, sd2 = (-7.0, -6.0), 1.0 # Mean and standard deviation for class 2 2d features
class1_input_data = np.random.normal(mean1, sd1, ((int)(0.5*n), m)) # (5000, 2) matrix having desired output label +1
class2_input_data = np.random.normal(mean2, sd2, ((int)(0.5*n), m)) # (5000, 2) matrix having desired output label -1

class1_desired_labels = np.full(((int)(0.5*n), 1), +1)
class2_desired_labels = np.full(((int)(0.5*n), 1), -1)

bias_input_term = np.full(((int)(0.5*n), 1), +1)

# Contatenate all into a single numpy nd-array
class1_data = np.concatenate((bias_input_term, class1_input_data, class1_desired_labels), axis = 1)
class2_data = np.concatenate((bias_input_term, class2_input_data, class2_desired_labels), axis = 1)
data = np.concatenate((class1_data, class2_data), axis = 0)

# Randomly shuffle the data
np.random.shuffle(data)

# Randomly initialize free parameters between -0.05 and 0.05
weights = 