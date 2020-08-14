# If geneticalgorithm is not installed on your machine, and using Jupyter Notebook or Google Colab, uncomment and run the below command
# !pip install geneticalgorithm

import time
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

# The objective function
def eggf(x):
    x1,x2 = x
    return x1**2 + x2**2 + 25*(np.sin(x1)**2) + np.sin(x2)**2

egg_fVals = []  # global variable to store function values for best iterations
fVals = []      # global variable to store egg_fVals from all the run to be used while plotting graph

# The function which is passed to the ga solver method to compute function values
def eggcrate(x):
    global egg_fVals
    fval = eggf(x)
    # If egg_fVals is empty, then store the first function value in it
    if not egg_fVals:
        print(f"{x}, {eggf(x)}")
        egg_fVals.append(fval)
    # Else if current function value is less than the last one and is also different, then store it in egg_fVals
    elif (fval < egg_fVals[-1] and fval != egg_fVals[-1]):
        # print("fval",fval)
        egg_fVals.append(fval)
        
    return eggf(x)

egg_lb = -2*np.pi   # the lower bound (same for both the variables)
egg_ub = 2*np.pi    # the upper bound (same for both the variables)
egg_bound = np.array([[egg_lb,egg_ub],[egg_lb,egg_ub]])
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
def egg_ga(maxRun):
    for i in range(maxRun):
        global egg_fVals
        global fVals
        egg_fVals = []
        egg_sol = ga(function=eggcrate, dimension=2, variable_type='real', variable_boundaries=egg_bound,
                            algorithm_parameters=algorithm_param)
        start_time = time.process_time()
        egg_sol.run()
        end_time = time.process_time()
        print("Execution Time: ", end_time - start_time)
        fVals.append(egg_fVals)
    plot_gaRosen(maxRun)
    return egg_sol

# Plotting the graph for best generations from each run on the same canvas
def plot_gaEgg(maxRun):
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Best Generations')
    ax1.set_ylabel('Function value')
    ax1.set_title('Egg Crate Function (linear view)', weight='bold')
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Best Generations')
    ax2.set_ylabel('Function value')
    ax2.set_title('Egg Crate Function (log view)', weight='bold')
    ax2.set_yscale("log")
    for i in range(maxRun):
        ax1.plot(fVals[i], marker='.', label=f"Run {i+1}")
        ax2.plot(fVals[i], marker='.', label=f"Run {i+1}")
    ax1.legend()
    ax2.legend()
    return