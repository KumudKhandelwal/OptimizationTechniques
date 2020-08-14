# If geneticalgorithm is not installed on your machine, and using Jupyter Notebook or Google Colab, uncomment and run the below command
# !pip install geneticalgorithm

import time
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

#   The objective function
def rosenf(x):
    x1,x2 = x
    return 100*(x2-x1**2)**2 + (1-x1)**2

rosen_fVals = []    # global variable to store function values for best iterations
rfVals = []         # global variable to store egg_fVals from all the run to be used while plotting graph

# The function which is passed to the ga solver method to compute function values
def rosenbrock(x):
    global rosen_fVals
    fval = rosenf(x)
    # If egg_fVals is empty, then store the first function value in it
    if not rosen_fVals:
        print(f"{x}, {rosenf(x)}")
        rosen_fVals.append(fval)
    # Else if current function value is less than the last one and is also different, then store it in egg_fVals
    elif (fval < rosen_fVals[-1] and fval != rosen_fVals[-1]):
        # print("fval",fval)
        rosen_fVals.append(fval)
    return rosenf(x)

rosen_lb = -5 # the lower bound (same for both the variables)
rosen_ub = 5  # the upper bound (same for both the variables)
rosen_bound = np.array([[rosen_lb,rosen_ub],[rosen_lb,rosen_ub]])
algorithm_param = {'max_num_iteration': None,
                    'population_size':100,
                    'mutation_probability':0.05,
                    'elit_ratio': 0.01,
                    'crossover_probability': 0.7,
                    'parents_portion': 0.3,
                    'crossover_type':'uniform',
                    'max_iteration_without_improv':200}
                    
# The ga solver method
# maxRun is the number of times you want to run the ga solver to compare results     
def rosen_ga(maxRun):
    for i in range(maxRun):
        global rosen_fVals
        global rfVals
        rosen_fVals = []
        rosen_sol = ga(function=rosenbrock, dimension=2, variable_type='real', variable_boundaries=rosen_bound,
                            algorithm_parameters=algorithm_param)
        start_time = time.process_time()
        rosen_sol.run()
        end_time = time.process_time()
        print("Execution Time: ", end_time - start_time)
        rfVals.append(rosen_fVals)
    plot_gaRosen(maxRun)
    return rosen_sol

# Plotting the graph for best generations from each run on the same canvas
def plot_gaRosen(maxRun):
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Best Generations')
    ax1.set_ylabel('Function value')
    ax1.set_title('Rosenbrock (linear view)', weight='bold')
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Best Generations')
    ax2.set_ylabel('Function value')
    ax2.set_title('Rosenbrock (log view)', weight='bold')
    ax2.set_yscale("log")
    for i in range(maxRun):
        ax1.plot(rfVals[i], marker='.', label=f"Run {i+1}")
        ax2.plot(rfVals[i], marker='.', label=f"Run {i+1}")
    ax1.legend()
    ax2.legend()
    return