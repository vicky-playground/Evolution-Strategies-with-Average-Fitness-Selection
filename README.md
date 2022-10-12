# CMA-ES using [average fitness selection strategy](https://ieeexplore.ieee.org/document/9870232)
[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](./LICENSE) [![PyPI - Downloads](https://img.shields.io/pypi/dw/cmaes)](https://pypistats.org/packages/cmaes)


Compare the performance of standard Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with CMA-ES using average fitness implementation.

## Concept
1. The baseline is to run some independent trials of CMA-ES, each starting from a different random start point.  
2. CMA-ES invokes a function to determine the fitness at the location x -- i.e. f(x).  Instead of x, we will use x1, x2, x3, ..., xn where n = 10 (for starters).  
3. Each of these n solutions is created by adding a number in the range of [-r,+r] that is drawn from a uniform distribution to each term of x.  For starters, r can be 10% of the search space, so r = 10.  Then, favg(x) = f(x1) + f(x2) + f(x3) + ... + f(xn) / n
4. After CMA-ES with average fitness finishes, we should use that point as the start point for standard CMA-ES with a small sigma (~0.25).  That should hopefully get close to 0.
5. Compare the result of the step 4 with the standard CMA-ES

## Result
OVerall, the result looks promising. Standard CMA-ES has an average performace around 50, and the average fitness has a final avergae performsnce less than 5
The result can be seen [here](https://www.notion.so/result-986cbbdcc2ee48cab02abc3f6d5d3f7c)

## Installation

Supported Python versions are 3.6 or later.

```
$ pip install cmaes
```

Or you can install via [conda-forge](https://anaconda.org/conda-forge/cmaes).

```
$ conda install -c conda-forge cmaes
```

## Usage

This library provides an "ask-and-tell" style interface. [1]

```python
import numpy as np
from cmaes import CMA
import random

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
```

This is the CMA-ES with average fitness selection.
1. Function cmaesFavg(n, dim, points): Put the arguments of trials, dimensions, and the number of neighbors into the function to calculate the values over evaluations in each trial (Note: the initial start points are random)
2. Function cmaesTest(point, dim): After finishing each trial of CMA-ES with average fitness, its final point is used for the standard CMA-ES with the default sigma(~0.25).
```python
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
```
```python
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
```

The Rastrigin function I used to be my objective function

```python
def rastrigin(x): # will put a list of points into the function
    val = 10*len(x)
    for i in range(len(x)):
        val += x[i]**2 - 10*np.cos(2*np.pi*x[i])
    return val   
```

**References:**
* [1] [cmaes 0.8.2] (https://pypi.org/project/cmaes/)
