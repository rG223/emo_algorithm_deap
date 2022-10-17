#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import sys
import pymop.factory
from moea_d import MOEAD

from deap import base
from deap import creator
from deap import tools


MAX_WEIGHT = 100

NGEN = 100
MU = 100
LAMBDA = 2
CXPB = 0.7
MUTPB = 0.2
UP = 1
LOW = 0

# Create random items and store them in the items' dictionary.
def attribute_solution(UP, LOW, size):
    return [random.uniform(low, up) for low, up in zip([LOW]*size, [UP]*size)]


def main(seed, size, objectives, p):
    random.seed(seed)

    # Create the item dictionary: item name is an integer, and value is
    # a (weight, value) 2-uple.
    problem = pymop.factory.get_problem(p, n_var=size, n_obj=objectives)
    creator.create("MultiObject_Min", base.Fitness, weights=(-1,)*objectives)
    creator.create("Individual", list, fitness=creator.MultiObject_Min)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("generate_solution", attribute_solution, UP, LOW, size)
    # Structure initializers
    toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.generate_solution)
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

    toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])

    toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=0, up=1)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=0, up=1, indpb=1 / size)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    stats = {}

    def lambda_factory(idx):
        return lambda ind: ind.fitness.values[idx]

    fitness_tags = ["Weight", "Value"]
    for tag in fitness_tags:
        s = tools.Statistics(key=lambda_factory(
            fitness_tags.index(tag)
        ))
        stats[tag] = s

    mstats = tools.MultiStatistics(**stats)
    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)

    ea = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=mstats, halloffame=hof, nr=LAMBDA)
    pop = ea.execute()

    return pop, stats, hof


if __name__ == "__main__":
    PROBLEM = "dtlz2"
    objectives = 3
    size = 30
    seed = 64

    pop, stats, hof = main(seed, size, objectives, PROBLEM)

    pop = [str(p) + " " + str(p.fitness.values) for p in pop]
    hof = [str(h) + " " + str(h.fitness.values) for h in hof]
