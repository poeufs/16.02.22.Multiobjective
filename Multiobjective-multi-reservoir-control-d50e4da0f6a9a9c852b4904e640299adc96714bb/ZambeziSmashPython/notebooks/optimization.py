import pandas as pd
import numpy as np
import os
os.chdir('../src')
from model_zambezi_OPT import model_zambezi
print('before ema_workbench')
from ema_workbench import (MultiprocessingEvaluator, ema_logging, RealParameter, ScalarOutcome, Constant,
                           Model)
#print('before model specification')
#from model_specification_ema import model_wrapper, ZambeziProblem

#import model_specification_ema
print('after imports')

ZambeziProblem = model_zambezi()

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
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE), #WHY MINIMIZE ?
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
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.optimize(nfe=500000 #250
                                     , searchover="levers", epsilons=[0.5] * len(model.outcomes))

    print(results)