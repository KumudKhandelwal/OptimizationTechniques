# Working Solution - Genralized form to run both rosenbrock and egg crate

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import time
from datetime import datetime


# Define the objective function
def rosenf(x):
    x1, x2 = x
    return 100*(x2-x1**2)**2 + (1-x1)**2

def eggcratef(x):
    x1, x2 = x
    return x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2)

# Define the bounds for rosenbrock
rosen_bounds = [(-5, 5), (-5, 5)]

# Define the bounds for egg Crate
eggcrate_bounds = [(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]

# global variables
iteration = 1
xvals = []
fvals = []
def rosen_callback(x):
    global iteration
    global xvals
    global fvals
    print("{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}".format(iteration, x[0], x[1], rosenf(x)))
    iteration += 1
    xvals.append(x)
    fvals.append(rosenf(x))

def eggcrate_callback(x):
    global iteration
    global xvals
    global fvals
    print("{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}".format(iteration, x[0], x[1], eggcratef(x)))
    iteration += 1
    xvals.append(x)
    fvals.append(eggcratef(x))

# function which computes the optimal solution for given number of runs
def optimalSolution(function, name, bounds, constraints, callback, maxRun):
    # variable that records optimal solution after each run
    runs = []
    # variable that records initial values of decision varibales x1...x7 for each run
    initial_x = []
    print(f"{name} function:")
    print(f"\n*  All the values are rounded-off upto 6 decimal places")
    print(f"** The Iter 0 represents the initial values\n")
    
    # Creating 2 figures (or canvas) for detailed plots
    fig, ax = plt.subplots(1,1)
    fig1, ax1 = plt.subplots(1,1)
    
    # Setting the labels and title for both the plots
    ax.set_xlabel(f"$Iteration#$")
    ax.set_ylabel(f"$Function$ $Value$")
    ax.set_title(f"{name} function (linear view)\n", fontweight='bold')
    ax1.set_xlabel(f"$Iteration#$")
    ax1.set_ylabel(f"$Function$ $Value$")
    ax1.set_title(f"{name} function (log view)\n", fontweight='bold')
    
    for i in range(maxRun):
        # defining the global variables so as to change them in local scope
        global iteration
        global xvals
        global fvals
        iteration = 0   # initializing the global varibale iteration = 0 for recording starting point
        xvals = []      # clearing the global xvals array for each run
        fvals = []      # clearing the global fvals array for each run
        
        # randomly generating all the 7 decision variables within the respective bounds
        x1 = np.random.uniform(bounds[0][0],bounds[0][1])
        x2 = np.random.uniform(bounds[1][0],bounds[1][1])

        # zipping both randomly generated decision variables in a single array variable 
        x0 = np.array([x1,x2])
        
        print(f"Run# {i+1}:")
        print("--------------------------------------------------------------")
        print("{0:4s}\t{1:9s}\t{2:9s}\t{3:9s}".format('Iteration', ' x1', ' x2', 'f(x)'))
        print("--------------------------------------------------------------")
        # Print the starting values
        print("{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}".format(iteration, x0[0], x0[1], function(x0)))
        # Setting the global iteration value to 1 as callback needs it internally
        iteration = 1
        # to plot the initial function value
        fvals.append(function(x0))
        
        # Run the optimize.minimize function to get optimal values
        start_time = time.process_time()     # start time before minimize function call
        sol = minimize(function, x0, bounds=bounds, callback=callback, options={'disp':True}, )
        end_time = time.process_time()       # end time after minimize function call
        print('Execution time: {}'.format(end_time - start_time))
        print("\n")

        # This will be used to randomply generate color code for each run
        rgb = np.random.rand(3,)
        
        # Plot the graph for each run on single canvas (figure)
        ax.plot(fvals, label=f"Run {i+1}", marker='.', color=rgb)
        ax1.plot(fvals, label=f"Run {i+1}", marker='.', color=rgb)
        ax1.set_yscale("log")

        # Placing the legends at desired location
        ax.legend(loc="lower right", bbox_to_anchor=(1.5, -0.0),borderaxespad=0., ncol=2)
        ax1.legend(loc="lower right", bbox_to_anchor=(1.5, -0.0),borderaxespad=0., ncol=2)
        
        # Displaying the grid within each figure
        ax.grid(color='lightgray', alpha=0.7)
        ax1.grid(color='lightgray', alpha=0.7)
        
        runs.append(sol)
        initial_x.append(x0)
    return np.asarray([runs, initial_x])

egg_solution, egg_x = optimalSolution(eggcratef, "Egg Crate", eggcrate_bounds, constraints=[], callback=eggcrate_callback, maxRun=10)
rosen_solution, rosen_x = optimalSolution(rosenf, "Rosenbrock", rosen_bounds, constraints=[], callback=rosen_callback,  maxRun=10)