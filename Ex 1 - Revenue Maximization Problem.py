# Complete solution using scipy

# Run the below command to resolve the display issue with mpld3. If this is not run, the interactive graph won't be displayed.
!pip install "git+https://github.com/javadba/mpld3@display_fix"

# if mpld3 is not install on machine and using Jupyter notebook or Google colab,
# use the below command to install mpld3 to plot interative graphs
!pip install mpld3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import mpld3

def objective(p):
    a = np.array([100, 150, 300]) 
    j = np.dot(p, demand(a,p))
    return -j

def demand(a,p):
    d = a * np.exp(-p/a)        # number of seats for a given p is computed using this formula
    # print("Demand:",d)
    return d

# Calculate the equality constraint based on number of seats
def eqConstraint(p, a, totalSeats):
    h = sum(demand(a,p)) - totalSeats
    return h

# Calculate the new prices after rounding off demand for seats (D1,D2,D2)
def optimalPricePerBucket(d):
    p = -a * np.log(d/a)
    return p

def optimalValues(a, p0, totalSeats):
    # Defining the equality constraint (Note: totalSeats will vary as per part 1 and part 2) 
    con = {'type': 'eq', 'fun':eqConstraint, 'args':[a, totalSeats]}
    solution = minimize(objective, p0,  method='SLSQP', bounds=None, constraints=con,  options={'disp':False})
    price = solution.x
    d = demand(a, price)
    print(f"    Demand for seats (original): D1 = {d[0]}, D2 = {d[1]}, and D3 = {d[2]}")
    print(f"    Fare buckets (in $): p1 = {price[0]}, p2 = {price[1]} and p3 = {price[2]},")
    print(f"    Maximum revenue: $ {-objective(price)}")
    return {'demand':d, 'price':price, 'revenue':-objective(price)}

def summary(a, p0, totalSeats):
    a1,a2,a3 = a
    p1,p2,p3 = p0
    print("~~~~~ Summary begins here ~~~~~\n\n")
    print(f"Fixed parameters: a1 = {a1}, a2 = {a2}, a3 = {a3}")
    print(f"Initial price bucket (in $): p1 = {p1}, p2 = {p2}, p3 = {p3}\n")
    print("Part 1: Total number of seats equals 150.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    part1 = optimalValues(a, p0, totalSeats)
    print("\n\nPart 2: Total number of seats are increased by 3")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    part2 = optimalValues(a, p0, totalSeats + 3)

    seatChange = part2['demand'] - part1['demand']
    priceChange = part2['price'] - part1['price']
    revenueChange = part2['revenue'] - part1['revenue']
    
    print(f"\n\nWe conclude that the company should make below amendments if it wants to add 3 more seats.")
    print(f"    Change in fare buckets (in $): p1 = {priceChange[0]}, p2 = {priceChange[1]} and p3 = {priceChange[2]}")
    print(f"    Change in seats: D1 = {seatChange[0]}, D2 = {seatChange[1]} and D3 = {seatChange[2]}")
    print(f"As a result, the company would increase its revenue by ${revenueChange}")
    print("\n\n~~~~~ Summary ends here ~~~~~")
    return plotFigure(part1, part2)

def plotFigure(part1, part2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = ['1','2']
    pl1 = ax.scatter(part1['price'],part1['demand'], c='r', label='Total seats: 150')
    pl2 = ax.scatter(part2['price'],part2['demand'], c='g', label='Total seats: 153')
    ax.set_xlabel('Fare per class')
    ax.set_ylabel('Demand per fare bucket')
    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()
    plt.show()
    return

# Starting point of the problem
a = np.array([100, 150, 300])
p0 = np.array([100.0, 100.0, 100.0])
totalSeats = 150
summary(a, p0, totalSeats)