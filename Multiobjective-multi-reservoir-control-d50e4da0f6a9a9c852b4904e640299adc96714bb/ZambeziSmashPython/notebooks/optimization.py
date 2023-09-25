import pandas as pd
import numpy as np
import os
os.chdir('../src')
from model_zambezi_OPT import ModelZambezi

print('before ema_workbench')
from ema_workbench import (MultiprocessingEvaluator, ema_logging, RealParameter, ScalarOutcome, Constant,
                           Model)
#print('before model specification')
#from model_specification_ema import model_wrapper, ZambeziProblem

#import model_specification_ema
print('after imports')

ZambeziProblem = ModelZambezi()

def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    #print('len kwargs is', len(kwargs)) = 230
    Hydropower, Environment, Irrigation = tuple(ZambeziProblem.evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation


#specify model
model = Model('zambeziproblem', function=model_wrapper)
print('after model definition')

#specify levers
model.levers = [RealParameter('v' + str(i), -1, 1) for i in range(ZambeziProblem.Nvar)]
print('after model.levers')

#specify outcomes
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE), # Minimize, because deficits
                  ScalarOutcome('Environment', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Irrigation', ScalarOutcome.MINIMIZE)]

print('after model outcomes')

if __name__ == '__main__':
    print('within main statement')
    ema_logging.log_to_stderr(ema_logging.INFO)

    # with MultiprocessingEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))

    # with SequentialEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))
    print('after ema logging')
    results = []
    with MultiprocessingEvaluator(model) as evaluator:
        for _ in range(5):
            result = evaluator.optimize(nfe=100 #500000 #250
                                     , searchover="levers", epsilons=[0.1] * len(model.outcomes)) # 0.05
        results.append(result)
        print(results)

    # Merge the 5 runs of the optimization
    from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem

   # os.chdir('../runs')

    problem = to_problem(model, searchover="levers")
    epsilons = [0.05] * len(model.outcomes)
    merged_archives = epsilon_nondominated(results, epsilons, problem)

    # save the results
    merged_archives.to_excel("merged_optimization_test.xlsx")


