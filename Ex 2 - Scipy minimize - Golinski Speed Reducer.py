# Working Solution

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import time

# Define the objective function
def gsr_objective(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    f = (0.7854*3.3333)*x1*x2**2*x3**2 + (0.7854*14.9334)*x1*x2**2*x3\
     - (0.7854*43.0934)*x1*x2**2 - 1.508*x1*x6**2 - 1.508*x1*x7**2\
      + 7.4777*x6**3 + 7.4777*x7**3 + 0.7854*x4*x6**2 + 0.7854*x5*x7**2
    return f

# Define all the constraint functions
def g1(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - 27/(x1*x2**2*x3)
def g2(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - 397.5/(x1*x2**2*x3**2)
def g3(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - 1.93*x4**3/(x2*x3*x6**4)
def g4(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - 1.93*x5**3/(x2*x3*x7**4)
def g5(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (745**2*x4**2/(x2**2 * x3**2) + 16.9*10**6 )**(0.5)/(110*x6**3)
def g6(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (745**2*x5**2/(x2**2 * x3**2) + 157.5*10**6)**(0.5)/(85*x7**3)
def g7(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - x2*x3/40
def g8(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - 5*x2/x1
def g9(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - x1/(12*x2)
def g24(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (1.5*x6 + 1.9)/x4
def g25(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (1.1*x7 + 1.9)/x5

# Define the nature of all the constraints (equality or inequality?)
gg1 = {'type': 'ineq', 'fun':g1}
gg2 = {'type': 'ineq', 'fun':g2}
gg3 = {'type': 'ineq', 'fun':g3}
gg4 = {'type': 'ineq', 'fun':g4}
gg5 = {'type': 'ineq', 'fun':g5}
gg6 = {'type': 'ineq', 'fun':g6}
gg7 = {'type': 'ineq', 'fun':g7}
gg8 = {'type': 'ineq', 'fun':g8}
gg9 = {'type': 'ineq', 'fun':g9}
gg24 = {'type': 'ineq', 'fun':g24}
gg25 = {'type': 'ineq', 'fun':g25}
gsr_constraints = [gg1, gg2, gg3, gg4, gg5, gg6, gg7, gg8, gg9, gg24, gg25]

# Define the bounds for x1, x2, x3, x4, x5, x6, x7
bound_x1 = (2.6, 3.6)
bound_x2 = (0.7, 0.8)
bound_x3 = (17, 28)
bound_x4 = (7.3, 8.3)
bound_x5 = (7.3, 8.3)
bound_x6 = (2.9, 3.9)
bound_x7 = (5.0, 5.9)
gsr_bounds = [bound_x1, bound_x2, bound_x3, bound_x4, bound_x5, bound_x6, bound_x7]

# global variables
iteration = 1
xvals = []
fvals = []
allConstraintNotFollowed = []
def gsr_callback(x):
    global iteration
    global xvals
    global fvals
    global allConstraintNotFollowed
    print("{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}\t{4: 3.6f}\t{5: 3.6f}\t{6: 3.6f}\t{7: 3.6f}\t{8: 3.6f}".format\
          (iteration, x[0], x[1], x[2], x[3], x[4], x[5], x[6], gsr_objective(x)))
    iteration += 1
    xvals.append(x)
    fvals.append(gsr_objective(x))
    # print(all(g(x)>0 for g in [g1,g2,g3,g4,g5,g6,g7,g8,g9,g24,g25]))
    feasible = lambda x: all(g(x)>0 for g in [g1,g2,g3,g4,g5,g6,g7,g8,g9,g24,g25])
    if not feasible(x):
        allConstraintNotFollowed.append(gsr_objective(x))

# function which computes the optimal solution for given number of runs
def optimalSolution(function, bounds, constraints, maxRun):
    # variable that records optimal solution after each run
    runs = []
    # variable that records initial values of decision varibales x1...x7 for each run
    initial_x = []
    print(f"\n*  All the values are rounded-off upto 6 decimal places")
    print(f"** The Iter 0 represents the initial values\n")
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Function Value")
    ax.set_title("Golinski Speed Reducer")
    # allConstraintNotFollowed = []
    for i in range(maxRun):
        # defining the global variables so as to change them in local scope
        global iteration
        global xvals
        global fvals
        iteration = 0   # initializing the global varibale iteration = 0 for recording starting point
        xvals = []      # clearing the global xvals array for each run
        fvals = []      # clearing the global fvals array for each run
        global allConstraintNotFollowed
        allConstraintNotFollowed = []
        # randomly generating all the 7 decision variables within the respective bounds
        x1 = np.random.uniform(2.6,3.6)
        x2 = np.random.uniform(0.7,0.8)
        x3 = np.random.randint(17,28)
        x4 = np.random.uniform(7.3,8.3)
        x5 = np.random.uniform(7.3,8.3)
        x6 = np.random.uniform(2.9,3.9)
        x7 = np.random.uniform(5.0,5.9)

        # zipping all the 7 randomly generated decision variables in a single array variable 
        x0 = np.array([x1,x2,x3,x4,x5,x6,x7])
        
        print(f"Run# {i+1}:")
        print("--------------------------------------------------------------\
----------------------------------------------------------------------")
        print("{0:4s}\t{1:9s}\t{2:9s}\t{3:9s}\t{4:9s}\t{5:9s}\t{6:9s}\t{7:9s}\t{8:9s}"\
        .format('Iter', ' X1', ' X2', ' X3', ' X4', ' X5', ' X6', 'X7', 'f(X)'))
        print("--------------------------------------------------------------\
----------------------------------------------------------------------")
        # Print the starting values
        print("{0:4d}\t{1: 3.6f}\t{2: 3.6f}\t{3: 3.6f}\t{4: 3.6f}\t{5: 3.6f}\t{6: 3.6f}\t{7: 3.6f}\t{8: 3.6f}".format\
          (iteration, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], function(x0)))
        # Setting the global iteration value to 1 as callback needs it internally
        iteration = 1
        # to plot the initial function value
        fvals.append(function(x0))
        
        # Run the optimize.minimize function to get optimal values
        start_time = time.process_time()
        sol = minimize(function, x0, bounds=bounds, constraints=constraints,
                       callback=gsr_callback, options={'disp':False}, )
        end_time = time.process_time()
        print("Execution time:",end_time - start_time)
        print("\n")

        # Plot the graph for each run on single canvas (figure)
        ax.plot(fvals, label=f"Run {i+1}", marker='.')
        ax.set_yscale("linear")
        ax.legend()
        
        runs.append(sol)
        initial_x.append(x0)
        # print("cons check:",allConstraintNotFollowed)
    return np.asarray([runs, initial_x])
    
gsr_solution, gsr_x = optimalSolution(gsr_objective, gsr_bounds, gsr_constraints, maxRun=10)