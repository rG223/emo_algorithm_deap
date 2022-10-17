from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import yaml


with open('NSGA2/base_nsga2.yaml', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

assert cfg['algorithm_name'] in ['NSGA2','NSGA3', 'MOEAD', 'RVEA', 'NSGA3']

problem = get_problem(cfg['problem_name'])

if cfg['algorithm_name'] == "NSGA2":
    algorithm = NSGA2(
        pop_size=cfg['pop_size']
    )
elif cfg['algorithm_name'] == "MOEAD":
    algorithm = MOEAD(
        pop_size=cfg['pop_size']
    )
elif cfg['algorithm_name'] == "RVEA":
    algorithm = RVEA(
        pop_size=cfg['pop_size']
    )
elif cfg['algorithm_name'] == "NSGA3":
    algorithm = NSGA3(
        pop_size=cfg['pop_size']
)


if __name__ == "__main__":
    res = minimize(problem,
                   algorithm,
                   ('n_gen', cfg['n_gen']),
                   seed=1,
                   verbose=True)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color='black', alpha=0.7)
    plot.add(res.F, facecolor='none', edgecolor='red')
    plot.show()
