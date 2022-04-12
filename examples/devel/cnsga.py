#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Useful for debugging
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline


# In[3]:


from xopt import nsga2, cnsga

import array
import time
from deap import base, tools, creator, algorithms
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np

import random


# # Test Problems

# In[20]:


NAME = "TNK"
BOUND_LOW, BOUND_UP = [0.0, 0.0], [3.14159, 3.14159]
WEIGHTS = [-1, -1]  # Minimize
NDIM = 2
N_CONSTRAINTS = 2


def TNK(individual):
    x1 = individual[0]
    x2 = individual[1]
    objectives = (x1, x2)
    constraints = (
        x1 ** 2 + x2 ** 2 - 1.0 - 0.1 * np.cos(16 * np.arctan2(x1, x2)),
        0.5 - (x1 - 0.5) ** 2 - (x2 - 0.5) ** 2,
    )
    return (objectives, constraints)


F = TNK
X_RANGE = [0, 1.4]
Y_RANGE = [0, 1.4]


def evaluate_TNK(settings):
    info = {"some": "info", "about": ["the", "run"]}

    ind = [settings["x1"], settings["x2"]]
    objectives, constraints = TNK(ind)

    output = {
        "y1": objectives[0],
        "y2": objectives[1],
        "c1": constraints[0],
        "c2": constraints[1],
    }

    return output


# # Setup Toolbox

# In[5]:


toolbox = nsga2.nsga2_toolbox(
    weights=WEIGHTS, n_constraints=N_CONSTRAINTS, bound_low=BOUND_LOW, bound_up=BOUND_UP
)


# In[6]:


# Register test proglem as toolbox.evaluate
toolbox.register("evaluate", F)


# # Test

# In[7]:


toolbox.population(n=10)


# In[8]:


creator.Individual([1, 2, 3])


# # Parallel method

# In[ ]:


# from concurrent.futures import ProcessPoolExecutor as PoolExecutor
# from concurrent.futures import ThreadPoolExecutor as PoolExecutor


# # Continuous NSGA-II, -III Loop

# In[9]:


# Wrap evaluate to return the input and output
def EVALUATE(vec):
    # sleep_time = random.random() *2
    # time.sleep(sleep_time)
    fit = F(vec)
    return vec, fit


# In[10]:


def cnsga(
    executor,
    toolbox=None,
    seed=None,
    evaluate_f=None,
    max_generations=2,
    population_size=4,
    crossover_probability=0.9,
    mutation_probability=1.0,
):
    """

    Continuous NSGA-II, NSGA-III

    Futures method, uses an executor as described in:
    https://www.python.org/dev/peps/pep-3148/

    Works with executors instantiated from:
       concurrent.futures.ProcessPoolExecutor
       concurrent.futures.ThreadPoolExecutor
       mpi4py.futures.MPIPoolExecutor
       dask.distributed.Client



    """
    random.seed(seed)
    MU = population_size
    CXPB = crossover_probability
    MUTPB = mutation_probability

    assert MU % 4 == 0, f"Population size (here {MU}) must be a multiple of 4"
    # Initial population
    pop = toolbox.population(n=MU)

    # Wrap evaluate to return the input and output
    def evaluate(vec):
        # sleep_time = random.random() *2
        # time.sleep(sleep_time)
        fit = evaluate_f(vec)
        return vec, fit

    # FIXME
    # evaluate = EVALUATE
    # evaluate = evaluate_f
    evaluate = toolbox.evaluate

    # function to reform individual
    def form_ind(vec, fit):
        ind = creator.Individual(vec)
        ind.fitness.values = fit[0]
        ind.fitness.cvalues = fit[1]
        ind.fitness.n_constraints = len(fit[1])
        return ind

    # Only allow vectors to be sent to evaluate
    def get_vec(ind):
        return array.array("d", [float(x) for x in ind])

    def get_vecs(inds):
        return [get_vec(ind) for ind in inds]

    # Individuals that need evaluating
    vecs = [get_vec(ind) for ind in pop if not ind.fitness.valid]

    futures = [executor.submit(evaluate, vec) for vec in vecs]

    # Clear pop
    pop = []
    for future in futures:
        vec, fit = future.result()
        ind = form_ind(vec, fit)
        pop.append(ind)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Make inital offspring to start the iteration
    vecs0 = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB))

    # Submit evaluation of initial population
    futures = [executor.submit(evaluate, vec) for vec in vecs0]

    generation = 0
    new_vecs = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB))
    new_offspring = []

    # Continuous loop
    done = False
    while not done:
        # Check the status of all futures
        for ix in range(len(futures)):

            # Examine a future
            fut = futures[ix]

            if fut.done():
                vec, fit = fut.result()
                ind = form_ind(vec, fit)
                new_offspring.append(ind)

                # Refill inputs
                if len(new_vecs) == 0:
                    pop = toolbox.select(pop + new_offspring, MU)
                    new_offspring = []
                    # New offspring
                    new_vecs = get_vecs(algorithms.varAnd(pop, toolbox, CXPB, MUTPB))

                    print(f"Generation {generation} completed")
                    generation += 1
                    if generation == max_generations:
                        done = True

                # Add new job for worker
                vec = new_vecs.pop()
                future = executor.submit(evaluate, vec)
                futures[ix] = future

        # Slow down polling. Needed for MPI to work well.
        time.sleep(0.001)

    # Cancel remaining jobs
    for future in futures:
        future.cancel()

    return pop


# with PoolExecutor() as executor:
#    pop = cnsga(executor, evaluate_f=F)


# # Parallel method

# In[16]:


# Dask distributed

# from dask.distributed import Client
# client = Client()
# client


# In[ ]:


# toolbox.register('evaluate', EVALUATE)
# pop = cnsga.cnsga(client, toolbox=toolbox, max_generations = 40, population_size=64)


# In[ ]:


# pop = cnsga(client, toolbox=toolbox, evaluate_f=F)


# # Recreate plots in Deb paper

# In[ ]:


import matplotlib.pyplot as plt


def plot_pop(pop):
    fig, ax = plt.subplots(figsize=(5, 5))

    front = np.array([ind.fitness.values for ind in pop])
    ax.scatter(front[:, 0], front[:, 1], color="blue")
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_aspect("auto")
    ax.set_title(NAME)


# plot_pop(pop)


# # MPI

# In[ ]:


from mpi4py.futures import MPIPoolExecutor as PoolExecutor


# In[15]:


from xopt import vocs_tools
from xopt.cnsga import cnsga as xcnsga

VOCS = vocs_tools.load_vocs("TNK_test.json")


# In[24]:


if __name__ == "__main__":
    with PoolExecutor() as executor:
        # pop = cnsga(executor, toolbox=toolbox, evaluate_f=EVALUATE)

        pop = xcnsga(
            executor,
            vocs=VOCS,
            evaluate_f=evaluate_TNK,
            max_generations=10,
            population_size=40,
        )
        plot_pop(pop)


# In[26]:


#!jupyter nbconvert --to script cnsga.ipynb


# In[ ]:
