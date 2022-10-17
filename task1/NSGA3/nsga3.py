from deap import base, tools, algorithms, creator
import numpy as np
import random
from deap.benchmarks.tools import hypervolume
import pymop.factory


def attribute_solution(UP, LOW, size):
    return [random.uniform(low, up) for low, up in zip([LOW]*size, [UP]*size)]


def ZDT1(ind):
    f1 = ind[0]
    g = 1 + 9/(len(ind)-1) * sum(ind[1:])
    f2 = 1 - np.sqrt(f1/g)
    return f1, f2


UP = 1
LOW = 0
size = 30
N = 200
N_gen = 250
cx_prob = 0.8
mut_prob = 0.2
random.seed(2022)
PROBLEM = "dtlz2"
problem = pymop.factory.get_problem(PROBLEM, n_var=size, n_obj=3)
# generate individual
creator.create("MultiObject_Min", base.Fitness, weights=(-1, -1, -1))
creator.create("Individual", list, fitness=creator.MultiObject_Min)

toolbox = base.Toolbox()
toolbox.register("generate_solution", attribute_solution, UP, LOW, size)
toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.generate_solution)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])

toolbox.register('select_to_mate', tools.selTournament, tournsize=2)
toolbox.register('select', tools.selNSGA3)
toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=0, up=1)
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=0, up=1, indpb=1/size)

# generate population

pop = toolbox.population(n=N)
fitnesses = toolbox.map(toolbox.evaluate, pop)
for fitness, ind in zip(fitnesses, pop):
    ind.fitness.values = fitness
ref_points = tools.uniform_reference_points(nobj=3, p=12)
pop = toolbox.select(pop, k=N, ref_points=ref_points)
hyper_volume = hypervolume(pop, [11.0, 11.0, 11.0])
print('gen: {}, hypervolume:{}'.format(0, hyper_volume))
# logging
stats = tools.Statistics()
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("mean", np.mean)
stats.register("std", np.std)
recode = stats.compile(pop)
logbook = tools.Logbook()
logbook.record(gen=0, **recode)

# generate new population
for gen in range(1, N_gen):
    parents = toolbox.select_to_mate(pop, k=N)
    offsprings = algorithms.varAnd(parents, toolbox, cxpb=cx_prob, mutpb=mut_prob)
    invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for fitness, ind in zip(fitnesses, invalid_ind):
        ind.fitness.values = fitness
    combined_pop = pop + invalid_ind

    pop = toolbox.select(combined_pop, k=N, ref_points=ref_points)
    recode = stats.compile(pop)
    logbook.record(gen=gen, **recode)
    if gen % 10 == 0:
        hyper_volume = hypervolume(pop, [11.0, 11.0, 11.0])
        print('***********iter:{}, hypervolume:{}'.format(gen, hyper_volume))

logbook.header = 'gen', 'max', 'min', 'mean', 'std'
print(logbook.stream)










