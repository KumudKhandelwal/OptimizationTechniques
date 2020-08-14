# If pyswarm is not installed on machine and using Jupyter notebook/ Google Colab, run the below command
!pip install pyswarm 

# Necessary imports
from pyswarm import pso
import numpy as np
import matplotlib.pyplot as plt
import time

def egg_objective(x):
    x1, x2 = x
    return x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2)

egg_funcVal = []
best = []
def eggcrate(x):
    global egg_funcVal
    global best

    func = egg_objective(x)
    
    if not egg_funcVal:
        egg_funcVal.append(func)
        print("Starting x: {}, function value: {}".format(x, egg_funcVal))
    
    if func < egg_funcVal[-1]:
        egg_funcVal.append(func)
    else:
        egg_funcVal.append(egg_funcVal[-1])

    if not best:
        best.append(egg_funcVal[-1])
        
    if func == egg_funcVal[-1] and best.count(func) == 0:
        best.append(egg_funcVal[-1])
        
    return func

eggcrate_lb = [-2*np.pi, -2*np.pi]
eggcrate_ub = [2*np.pi, 2*np.pi]

def optimalResult(function, name, lb, ub, maxRun, swarmsize=100, omega=0.5, phip=0.5, phig=0.5):
    fig = plt.figure(figsize=(10,10))
    for i in range(maxRun):
        global egg_funcVal
        global best
        
        egg_funcVal = []
        best = []
                
        print(f"Run# {i+1}:")
        
        start_time = time.process_time()
        xopt, fopt = pso(function,
                         lb, ub, debug=True, minstep=1e-6, minfunc=1e-6,
                         swarmsize=swarmsize,
                         omega=omega,
                         phip=phip,
                         phig=phig)
        end_time = time.process_time()
        print("final xopt:",xopt," final fopt:",fopt)
        print("Executon time:",end_time - start_time)
        ax1 = fig.add_subplot(221)
        ax1.plot(egg_funcVal, marker='', label=f'Run {i+1}')
        ax1.plot(0, egg_funcVal[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax1.plot(len(egg_funcVal), egg_funcVal[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax1.grid(color='lightgray', alpha=0.7)
        ax1.legend()
        ax1.set_xlabel("Swarm position change")
        ax1.set_ylabel("Function value")
        ax1.set_title(f"{name} function \n(complete cycle, linear view)", fontweight='bold')

        ax2 = fig.add_subplot(222)
        ax2.plot(egg_funcVal, label=f'Run {i+1}')
        ax2.plot(0, egg_funcVal[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax2.plot(len(egg_funcVal), egg_funcVal[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax2.grid(color='lightgray', alpha=0.7)
        ax2.set_yscale("log")
        ax2.set_xlabel("Swarm position change")
        ax2.set_ylabel("Function value")
        ax2.set_title(f"{name} function \n(complete cycle, log view)", fontweight='bold')
        ax2.legend()
        print("\n\n")
        ax3 = fig.add_subplot(223)
        ax3.plot(best, label=f'Run {i+1}')
        ax3.plot(0, best[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax3.plot(len(best)-1, best[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax3.grid(color='lightgray', alpha=0.7)
        ax3.set_xlabel("Swarm position change")
        ax3.set_ylabel("Objective function value")
        ax3.set_title(f"{name} function \n(Best cycle, linear view)", fontweight='bold')
        ax3.legend()

        ax4 = fig.add_subplot(224)
        ax4.plot(best, label=f'Run {i+1}')
        ax4.plot(0, best[0], marker='o', color=plt.gca().lines[-1].get_color())
        ax4.plot(len(best)-1, best[-1], marker='o', color=plt.gca().lines[-1].get_color())
        ax4.set_yscale("log")
        ax4.grid(color='lightgray', alpha=0.7)
        ax4.set_xlabel("Swarm position change")
        ax4.set_ylabel("Objective function value")
        ax4.set_title(f"{name} function \n(Best cycle, log view)", fontweight='bold')
        ax4.legend()

        plt.tight_layout()
    return
    
# eggcrate contains the objective function and the callback methods implemented to store optimal function values at best/ new best iterations
# "Rosenbrock" is passed to the name name parameter which will be printed out while plotting is done
# rosenbrock_lb is the list that has all the lower bounds
# rosenbrock_ub is the list that has all the upper bounds
# maxRun defines the number of times in loop one wants to run the same code to observe the behaviour of method implemented
# swarmsize, omega, phip, and phig are passed as arguments to make them dynamic
optimalResult(eggcrate, "Egg Crate", eggcrate_lb, eggcrate_ub, maxRun=10, swarmsize=100, omega=0.5, phip=0.5, phig=0.5)