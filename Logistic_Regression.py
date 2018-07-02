import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import loadtxt
import math


#plots graph of cost against iterations
def plot_graph(costs):

    x=[]

    #put no of itrs in x
    for i in range(len(costs)):
        x.append(i+1)

    plt.plot(x,costs)
    plt.show()



def map_features(data, degree):

    mapped_2 = []

    length = len(data)

    for i in range(length):

        mapped = []

        x1 = data[i,0]
        x2 = data[i,1]

        for j in range(1, degree + 1):
            for k in range(j + 1):
                r = (x1 ** (j - k)) * (x2 ** k)
                mapped.append(r)

        mapped.append(data[i, 2])
        mapped_2.append(mapped)


    return (np.array(mapped_2))


def calculate_error(thetas, data, reg_const):

    # add row of 1s to start of data set for dummy x0 vars
    col = np.ones((len(data), 1), dtype=float)       # gen column of 1's with as many rows as data
    data = np.hstack((col, data))                    # combine the two np arrays

    length = len(data)
    no_of_features = len(data[0]) - 1
    value = 0
    sum_of_squares = 0

    for i in range(length):
        y = data[i,no_of_features]
        pred_val = sigmoid(thetas, data, i)

        value += -y * np.log(pred_val) - (1-y) * np.log(1 - pred_val)

    for j in range(1, no_of_features):
        sum_of_squares += thetas[j] * thetas[j]

    reg_term = (reg_const/(2*length)) * sum_of_squares

    error = value/length
    error = error + reg_term

    return error



def predict_prob(thetas, features):

    no_of_terms = len(thetas)
    z = 0
    for i in range(no_of_terms):
        z += features[i] * thetas[i]

    prob = 1 / (1 + math.exp(-1.0 * z))
    return prob


def sigmoid(current_thetas, data, index):

    length = len(data)
    no_of_gradients = len(data[0]) - 1
    val = 0

    for j in range(no_of_gradients):
        val += current_thetas[j] * data[index, j]

    return 1 / (1 + math.exp(-1.0 * val))


def gradient_step(current_thetas, data, learning_rate, reg_const):

    gradients = []
    new_thetas = []
    no_of_gradients = len(data[0]) - 1
    length = len(data)
    total_error = 0
    sum_of_squares = 0

    #init gradients to 0
    for i in range(no_of_gradients):
        gradients.append(0)

    for i in range(length):

        # calc predicted val
        pred_val = sigmoid(current_thetas, data, i)

        for j in range(no_of_gradients):

            x = data[i, j]
            y = data[i, no_of_gradients]

            gradients[j] += (((pred_val) - y)*x) / length

        total_error += -y * np.log(pred_val) - (1 - y) * np.log(1 - pred_val)            #calculate cost at each itr for the plot

    new_thetas.append(current_thetas[0] - (learning_rate * gradients[0]))

    for j in range(1, no_of_gradients):
        new_thetas.append(current_thetas[j]*(1 - learning_rate*(reg_const/length)) - (learning_rate * gradients[j]))

    for j in range(1, no_of_gradients):                            #added term fro regularized regression cost
        sum_of_squares += current_thetas[j] * current_thetas[j]

    reg_term = (reg_const/(2*length)) * sum_of_squares

    cost = total_error/length
    cost = cost + reg_term
    return new_thetas, cost


def logistic_regression(data, num_iterations, learning_rate, reg_const):

    costs = [0]*num_iterations                         #init list with num_itr 0s
    
     # add row of 1s to start of data set for dummy x0 vars
    col = np.ones((len(data), 1), dtype=float)      # gen column of 1's with as many rows as data
    data = np.hstack((col, data))                   # combine the two np arrays


    #init weights
    thetas = []
    for i in range(len(data[0])-1):                 #last column is target
        thetas.append(0)

    #run step fucntion
    for i in range(num_iterations):
        thetas, costs[i] = gradient_step(thetas, data, learning_rate, reg_const)

    plot_graph(costs)
    return thetas


