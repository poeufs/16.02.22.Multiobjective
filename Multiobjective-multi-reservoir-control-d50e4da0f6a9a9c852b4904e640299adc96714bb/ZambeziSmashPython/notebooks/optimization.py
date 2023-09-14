import pandas as pd
import numpy as np
import os
os.chdir('../src')
from model_zambezi_OPT import model_zambezi

from platypus import NSGAII, Problem, Real
from ema_workbench import MultiprocessingEvaluator, ema_logging
ZambeziProblem = model_zambezi()

problem = Problem(ZambeziProblem.Nvar, ZambeziProblem.Nobj)
problem.types[:] = Real(0, 1)
problem.function = ZambeziProblem.evaluate

algorithm = NSGAII(problem=problem, population_size=20)
algorithm.run(100)

objectives_outcome = dict()
for i, column_name in enumerate(['Hydropower','Environment','Irrigation']):
    objectives_outcome[column_name] = [s.objectives[i] for s in algorithm.result]

objectives_df = pd.DataFrame(objectives_outcome)

print(objectives_df)
from src.various_plots import parallel_plots
parallel_plots(objectives_df)

def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    Hydropower, Environment, Irrigation = tuple(ZambeziProblem.evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation

from ema_workbench import (RealParameter, ScalarOutcome, Constant,
                           Model)

model = Model('zambeziproblem', function=model_wrapper)

model.levers = [RealParameter('v' + str(i), -1, 1) for i in range(ZambeziProblem.Nvar)]

#specify outcomes
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Environment', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Irrigation', ScalarOutcome.MINIMIZE)]


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    # with MultiprocessingEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))

    # with SequentialEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))

    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))

    print(results)