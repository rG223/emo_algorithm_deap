import math
import numpy as np
from deap import creator, base, tools, algorithms
import random
from deap.benchmarks.tools import hypervolume
import matplotlib.pyplot as plt
import yaml

with open('base_nsga2.yaml', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

print('CFG:', cfg)

# parameter setting

NDIM = cfg['NDIM']
N_pop = cfg['pop_size']
max_gen = cfg['n_gen']
cx_prob = cfg['cx_prob']
mutate_prob = cfg['mutate_prob']

domain = {'zdt1': [[0]*NDIM, [1]*NDIM],
          'zdt2': [[0]*NDIM, [1]*NDIM],
          'zdt3': [[0]*NDIM, [1]*NDIM],
          'zdt4': [[0]+[-5]*9, [1] + [5]*9]}
# define a smallest unit, i.e., individual.
creator.create("MultiObjective_Min", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.MultiObjective_Min)

# define individuals' features in domain with read-encoding.
def gen_ind_fea(low, up):
    return [random.uniform(l, u) for l, u in zip(low, up)]

# Given the LOW and UP of problem to generate individual
LOW = domain[cfg['problem_name']][0]
UP = domain[cfg['problem_name']][1]
toolbox = base.Toolbox()
toolbox.register("attributed_fea", gen_ind_fea, LOW, UP)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attributed_fea)

# create population
toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
pop = toolbox.Population(n=N_pop)

# define the solving problem, e.g., ZDT1
def ZDT1(ind):
    f1 = ind[0]
    g = 1 + 9/(len(ind)-1) * sum(ind[1:])
    f2 = 1 - math.sqrt(f1/g)
    return f1, f2

# register the evaluating function and some other important function.
toolbox.register("evaluate", ZDT1)
toolbox.register("select_gen", tools.selTournamentDCD)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=LOW, up=UP)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=LOW, up=UP, indpb=1.0/NDIM)

# logging
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("max", np.max)
stats.register("min", np.min)

logbook = tools.Logbook()

# GA loop
fitness = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit

logbook.header = ['gen', 'avg', 'std', 'max', 'min']
record = stats.compile(pop)
logbook.record(gen=0, **record)
# get pareto front
pop = toolbox.select(pop, k=N_pop)
# offsprings
offsprings = toolbox.select_gen(pop, N_pop)
offsprings = algorithms.varAnd(offsprings, toolbox, cx_prob, mutate_prob)
best_hv = 0
# begin the second iter...
for iter in range(1, max_gen):

    fitness = map(toolbox.evaluate, offsprings)
    for ind, fit in zip(offsprings, fitness):
        ind.fitness.values = fit
    combined_pop = pop + offsprings
    pop = toolbox.select(combined_pop, k=N_pop)

    offsprings = toolbox.select_gen(pop, k=N_pop)
    offsprings = algorithms.varAnd(offsprings, toolbox, cx_prob, mutate_prob)
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    if iter % 10 == 0:
        hv = hypervolume(pop, [11.0, 11.0])
        print('***************iter:{}*****************'.format(iter))
        if hv > best_hv:
            print('HV indicator:{:.4f}, improved: {:.4f}'.format(hv, hv - best_hv))
            best_hv = hv
        print(record)


# stream
hv = hypervolume(pop, [11.0, 11.0])
bestInd = tools.selBest(pop, 1)[0]
bestFit = bestInd.fitness.values
print('Best solution:', bestInd)
print('min of the function:', bestFit)
print('HV: ', hv)

front = tools.emo.sortNondominated(pop, len(pop))[0]
# visual
for ind in front:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
plt.xlabel('f1')
plt.ylabel('f2')
# plt.tight_layout()
plt.show()





















