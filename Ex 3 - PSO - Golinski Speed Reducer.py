# If pyswarm is not installed on machine and using Jupyter notebook/ Google Colab, run the below command
!pip install pyswarm

from pyswarm import pso
import numpy as np
import matplotlib.pyplot as plt
import time

# Golinksi Speed Reducer

def gsr_objective(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    f = (0.7854*3.3333)*x1*x2**2*x3**2 + (0.7854*14.9334)*x1*x2**2*x3\
     - (0.7854*43.0934)*x1*x2**2 - 1.508*x1*x6**2 - 1.508*x1*x7**2\
      + 7.4777*x6**3 + 7.4777*x7**3 + 0.7854*x4*x6**2 + 0.7854*x5*x7**2
    return f

gsr_lb = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]
gsr_ub = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.9]

def gsr_constraint_function(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    g1 = 1 - 27/(x1*x2**2*x3)
    g2 = 1 - 397.5/(x1*x2**2*x3**2)
    g3 = 1 - 1.93*x4**3/(x2*x3*x6**4)
    g4 = 1 - 1.93*x5**3/(x2*x3*x7**4)
    g5 = 1 - (745**2*x4**2/(x2**2*x3**2) + 16.9*10**6)**(0.5)/(110*x6**3)
    g6 = 1 - (745**2*x5**2/(x2**2*x3**2) + 157.5*10**6)**(0.5)/(85*x7**3)
    g7 = 1 - (x2*x3)/40
    g8 = 1 - 5*x2/x1
    g9 = 1 - x1/(12*x2)
    g24 = 1 - (1.5*x6 + 1.9)/x4
    g25 = 1 - (1.1*x7 + 1.9)/x5
    gsr_constraints = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g24, g25])
    return gsr_constraints

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
    return 1 - (745**2*x4**2/(x2**2*x3**2) + 16.9*10**6)**(0.5)/(110*x6**3)
def g6(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (745**2*x5**2/(x2**2*x3**2) + 157.5*10**6)**(0.5)/(85*x7**3)
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
    return 1 - (1.5*x6 + 1.9)*x4**(-1)
def g25(x):
    x1,x2,x3,x4,x5,x6,x7 = x
    return 1 - (1.1*x7 + 1.9)*x5**(-1)

gsr_funcVal = []    # list to store all the function values from all the iterations (complete cycle)
gsr_best = []       # list to store all the function values from best iterations (best cycle)

def gsr(x):
    global gsr_funcVal
    global gsr_best
    
    func = gsr_objective(x)
    
    # If gsr_funcVal is empty, then append the first function value to it
    if not gsr_funcVal:
        gsr_funcVal.append(func)
        print("Starting x: {}, function value: {}".format(x, gsr_funcVal))
    
    # If the current function value is less than the last one and if all the constraints are satisfied, then only append the current function value
    if func < gsr_funcVal[-1] and np.all(gsr_constraint_function(x) >= 0):
        gsr_funcVal.append(func)
    else:
        gsr_funcVal.append(gsr_funcVal[-1])
    
    # If the best value array is empty, append the first function value 
    if not gsr_best:
        # gsr_best.append(gsr_funcVal[-1])
        gsr_best.append(func)
    
    # If the current function value is same as the last value, then append it to the best list
    if func == gsr_funcVal[-1] and gsr_best.count(func) == 0:
        gsr_best.append(gsr_funcVal[-1])

    return func

def gsr_optimalResult(function, name, lb, ub, maxRun=10, maxiter=1000, swarmsize=100, omega=0.5, phip=0.5, phig=0.5):
    fig = plt.figure(figsize=(10,6))
    for i in range(maxRun):
        global gsr_funcVal
        global gsr_best
        
        gsr_funcVal = []
        gsr_best = []
                
        print(f"\nRun# {i+1}:")
        
        start_time = time.process_time()
        xopt, fopt = xopt, fopt = pso(function,
                 gsr_lb,
                 gsr_ub,
                 f_ieqcons=gsr_constraint_function,
                 maxiter=maxiter,
                 swarmsize=swarmsize,
                 debug=False,
                 minstep=1e-5,
                 omega=omega,
                 phig=phig,
                 phip=phip,
                 minfunc=1e-05,
                 )
        end_time = time.process_time()
        print("final xopt:",xopt," final fopt:",fopt)
        print("Executon time:",end_time - start_time)
        ax1 = fig.add_subplot(121)
        ax1.plot(gsr_funcVal, marker='', label=f'Run {i+1}')
        ax1.plot(0, gsr_funcVal[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax1.plot(len(gsr_funcVal), gsr_funcVal[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax1.grid(color='lightgray', alpha=0.7)
        ax1.legend()
        ax1.set_xlabel("Swarm position change")
        ax1.set_ylabel("Objective function value")
        ax1.set_title(f"{name} function \n(complete cycle, linear view)", fontweight='bold')
        
        ax2 = fig.add_subplot(122)
        ax2.plot(gsr_best, marker="", label=f'Run {i+1}')
        ax2.plot(0, gsr_best[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax2.plot(len(gsr_best)-1, gsr_best[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax2.grid(color='lightgray', alpha=0.7)
        ax2.set_xlabel("Best swarm position change")
        ax2.set_ylabel("Objective function value")
        ax2.set_title(f"{name} function \n(Best cycle, linear view)", fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        
    return
    
gsr_optimalResult(gsr, "Golinksi Speed Reducer", gsr_lb, gsr_ub, maxRun=10, maxiter=200, swarmsize=200, omega=0.55, phip=1.50, phig=1.52)