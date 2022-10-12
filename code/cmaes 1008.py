import math
import numpy as np
from cmaes import CMA
import random
import matplotlib.pyplot as plt

# =============================================================================
# plot the fitness and sigma
# =============================================================================
def drawPlot(values, sigma):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.xlabel('number of evaluations')
    plt.ylabel('fitness value')
    plt.plot(list(range(len(values))), values)
    plt.title(label='The first trial',horizontalalignment='center',)
    
    plt.subplot(1, 2, 2) # index 2
    plt.xlabel('number of evaluations')
    plt.ylabel('sigma')
    plt.plot(list(range(len(sigma))), sigma)
    plt.title(label='The first trial',horizontalalignment='center',)
    plt.show()

# =============================================================================
# Objective functions: Rastrigin function
# =============================================================================
def rastrigin(x): # will put a list of points into the function
    val = 10*len(x)
    for i in range(len(x)):
        val += x[i]**2 - 10*np.cos(2*np.pi*x[i])
    return val   

# =============================================================================
# CMA-ES 
# source: https://pypi.org/project/cmaes/
# =============================================================================
def cmaes(n, dim):# n: trials, dim:dimensions
    countFinal =[] # store the last average fitness of each trial    
    
    for i in range(n): 
        holdValue, holdSigma = [], [] # store all fitness values and sigma in this trial
        bounds = np.array([[-5.12, 5.12]])
        for _ in range(dim-1):
            bounds = np.append(bounds,np.array([[-5.12, 5.12]]))
        bounds = bounds.reshape(dim,2)
        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

        mean = lower_bounds + (np.random.rand(1) * (upper_bounds - lower_bounds)) # original point for CMAES
        sigma = 5.12 * 2 / 5  # 1/5 of the domain width
        optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)
        print("\nevals    f(x)")
        print("========  ==========")
        evals = 0
        
        while True:
            generation = 1
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                #print(f"solution generated: {x}")
                value = rastrigin(x)
                holdValue.append(value)  
                holdSigma.append(optimizer._sigma)
                #print(f"values: {value}")
                solutions.append((x, value))        
                #print(f"solutions:{solutions}")
                evals += 1
                if evals % 3000 == 0:
                    print(f"{evals:5d}  {value:10.5f}")
            optimizer.tell(solutions)

            if optimizer.should_stop():
                #popsize multiplied by 2 (or 3) before each restart.
                popsize = optimizer.population_size * 2
                mean = lower_bounds + (np.random.rand(1) * (upper_bounds - lower_bounds))
                optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
                #print(f"Restart CMA-ES with popsize={popsize}")
                break
            generation += 1
            
        countFinal.append(holdValue[len(holdValue)-1])   
        if i == (n-1):
            print("========")
            print(f"mean fitness:{np.mean(countFinal):10.5f}")
            print(f"best fitness of {n} runs:{np.min(countFinal):10.5f}")
#        if (i==0):  # draw the plot of the first CMA-ES trial 
#            drawPlot(holdValue, holdSigma)
        

def cmaesTest(point, dim):
    holdValue, holdSigma = [], [] # store all fitness values and sigma in this trial
    bounds = np.array([[-5.12, 5.12]])
    for _ in range(dim-1): 
        bounds = np.append(bounds,np.array([[-5.12, 5.12]]))
    bounds = bounds.reshape(dim,2)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    optimizer = CMA(mean=point, sigma=0.25, bounds=bounds, seed=0)
    
    print("\nt-evals    f(x)")
    print("========  ==========")
    evals = 0

    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = rastrigin(x)
            holdValue.append(value)  
            holdSigma.append(optimizer._sigma)
            solutions.append((x, value))        
            evals += 1
            if evals % 300 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            #popsize multiplied by 2 (or 3) before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(1) * (upper_bounds - lower_bounds))
            optimizer = CMA(mean=mean, sigma=0.25, population_size=popsize)
            #print(f"Restart CMA-ES with popsize={popsize}")
            break
        
    print(f"best fitness of cmaesTest:{np.min(holdValue[len(holdValue)-1]):10.5f}")
               
# =============================================================================
# CMA-ES using average fitness selection 
# =============================================================================
def cmaesFavg(n, dim, points): # n: trials; dim:dimensions; points: the number of neighbors

    countFinal =[] # store the last average fitness of each trial

    for k in range(n):
        bounds = np.array([[-5.12, 5.12]])
        for _ in range(dim-1):
            bounds = np.append(bounds,np.array([[-5.12, 5.12]]))
        bounds = bounds.reshape(dim,2)
        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
        
        mean = lower_bounds + (np.random.rand(1) * (upper_bounds - lower_bounds)) # random start point
        sigma = 5.12 * 2 / 5  # 1/5 of the domain width
        optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)
        
        print("\navg-evals    f(x)")
        print("=========  ==========")
        evals = 0  # number of function evaluations
        holdValue, holdSigma, finalPoint = [], [], [] # store all fitness values, sigma, final point in each trial
        
        while True:
            solutions = [] 
            for _ in range(optimizer.population_size):
                x = optimizer.ask() # get list of new solutions
                #print(f"solution generated: {x}")
                neighbors, numbers = [], points
                values = [] # store the neighboring value 
                
                #generate a random point within a range of [-0.5,0.5]
                # check if a point is within the bounds of the search
                # if not, the point should be regenerated 
                for _ in range(numbers):
                    nb = []
                    for j in range(dim):
                        xj = 0            
                        while True: 
                            if x[j] == 5.12:
                                xj = x[j] + random.uniform(-0.5, 0)
                            elif x[j] == -5.12:
                                xj = x[j] + random.uniform(0, 0.5)
                            else:
                                xj = x[j] + random.uniform(-0.5, 0.5)
                            if  np.all(-5.12 <= xj) and np.all(xj<= 5.12):
                                break 
                        nb.append(xj)  
                    neighbors.append(nb)
                # print(f"neighbors: {neighbors}")       
                # calculate the fitness of these neighbors
                for i in range(numbers): 
                    #print(f"neighbors[{i}][0]:{neighbors[i][0]},neighbors[{i}][1]:{neighbors[i][1]}")
                    value = rastrigin(neighbors[i]) 
                    values.append(value) 
                #print(f"values: {values}; len: {len(values)}")
                avg = sum(values)/len(values) # the average fitness of the neighbors
                #print(f"avg: {avg}")
                holdValue.append(avg) 
                holdSigma.append(optimizer._sigma)
                solutions.append((x,avg)) #use favg for its final fitness and start normal CMA-ES from the final x location
                #print(f"solutions:{solutions}")
                evals += 1
                if evals % 30000 == 0:  
                    print(f"{evals:5d}  {value:10.5f}")
            #print(f"solutions-2:{solutions}")
            optimizer.tell(solutions)
            if optimizer.should_stop():
                #print(f"solutions:{solutions}")
                finalPoint = solutions[-1][0]
                #print(f"finalPoint:{finalPoint}")
                cmaesTest(finalPoint,dim)
                #popsize multiplied by 2 (or 3) before each restart.
                popsize = optimizer.population_size * 2
                mean = lower_bounds + (np.random.rand(1) * (upper_bounds - lower_bounds))
                optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
                #print(f"Restart CMA-ES with popsize={popsize}")
                break
        print(f"#{k} cmaesFavg evals: {evals}")
        countFinal.append(holdValue[len(holdValue)-1])
        
        if k == (n-1):
            print("========")
            print(f"mean fitness:{np.mean(countFinal):10.5f}")
            print(f"best fitness of {n} runs:{np.min(countFinal):10.5f}")
        #if (k==0):  # draw the plot of the first CMA-ES trial 
        #    drawPlot(holdValue, holdSigma)

            
cmaes(30,30)
cmaesFavg(30,5,10)