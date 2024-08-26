import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tqdm import tqdm

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Train time should be O(N * (2 ^ d) * M)
# Testing time should be O(d * N)

# Function to create fake data (take inspiration from usage.py)
# Real input real output
N_max = [1, 5, 10, 30, 60]
M_max = [1, 5, 10, 20]

def get_data(type, N, M):
    np.random.seed(42)
    
    if type == 'real_input_real_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif type == 'real_input_discrete_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(0, 2, N))
        return X, y
    elif type == 'discrete_input_real_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif type == 'discrete_input_discrete_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randint(0, 2, N))
        return X, y
    else:
        raise ValueError("Invalid type")

def get_time_real_input_real_output(N, M, type):
    X, y = get_data(type, N, M)
    training_times = []
    testing_times = []
    for _ in tqdm(range(num_average_time)):
        tree = DecisionTree(criterion='information_gain')
        start = time.process_time()
        tree.fit(X, y)
        end = time.process_time()
        training_times.append(end - start)
        start = time.process_time()
        tree.predict(X)
        end = time.process_time()
        testing_times.append(end - start)

    return np.mean(training_times), np.mean(testing_times)

def plot_twin_axis_graph(x, y1, y2, title, xlabel):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_title(title)
    ax1.set_ylabel('time')
    ax1.set_xlabel(xlabel, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('time', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(["Training Time", "Testing Time"])
    # fig.show()
    fig.savefig("time_complexity_plots/" + title + ".png")


def plot_graph(N_vals, M_vals, fn, type):
    print("Plotting graph for", type)
    plt.title(type + " wrt N")
    plt.xlabel('N')
    plt.ylabel('Time')

    training_times = []
    testing_times = []

    for i in N_vals:
        training_time, testing_time = fn(i, 5, type)
        training_times.append(training_time)
        testing_times.append(testing_time)
        
    # Plot the graph in twin axis
    plot_twin_axis_graph(N_vals, training_times, testing_times, type + " wrt N", 'N')

    training_times = []
    testing_times = []
    
    for i in M_vals:
        training_time, testing_time = fn(20, i, type)
        training_times.append(training_time)
        testing_times.append(testing_time)
        
    plot_twin_axis_graph(M_vals, training_times, testing_times, type + " wrt M", 'M')
    
plot_graph(N_max, M_max, get_time_real_input_real_output, "real_input_real_output")
plot_graph(N_max, M_max, get_time_real_input_real_output, "real_input_discrete_output")
plot_graph(N_max, M_max, get_time_real_input_real_output, "discrete_input_real_output")
plot_graph(N_max, M_max, get_time_real_input_real_output, "discrete_input_discrete_output")
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
