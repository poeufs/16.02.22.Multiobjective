import pandas as pd
import numpy as np
import os
from model_zambezi_OPT import model_zambezi
from platypus import NSGAII, Problem, Real
os.chdir('../src')

# Initialize the problem
ZambeziProblem = model_zambezi()

problem = Problem(ZambeziProblem.Nvar, ZambeziProblem.Nobj)
problem.types[:] = Real(0, 1)
problem.function = ZambeziProblem.evaluate

if __name__ == '__main__':
    algorithm = NSGAII(problem=problem, population_size=20)
    algorithm.run(100)
    # algorithm.run(1)

    objectives_outcome = dict()
    for i, column_name in enumerate(['Hydropower', 'Environment', 'Irrigation']):
        objectives_outcome[column_name] = [s.objectives[i] for s in algorithm.result]

    objectives_df = pd.DataFrame(objectives_outcome)

    print("objectives:", objectives_df)
    from various_plots import parallel_plots

    parallel_plots(objectives_df)
