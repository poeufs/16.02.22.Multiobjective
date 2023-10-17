import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model_zambezi_OPT import ModelZambezi
from ema_workbench import (MultiprocessingEvaluator, ema_logging, RealParameter, ScalarOutcome, Constant,
                           Model, HypervolumeMetric,)
from ema_workbench.em_framework.optimization import (GenerationalBorg, epsilon_nondominated, to_problem, ArchiveLogger,
                                                     EpsilonProgress)
#from ema_workbench import (GenerationalDistanceMetric, EpsilonIndicatorMetric, InvertedGenerationalDistanceMetric,
                            #SpacingMetric,)
os.chdir('../src')
# print('after imports')

ZambeziProblem = ModelZambezi()


def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    # print('len kwargs is', len(kwargs)) = 230
    Hydropower, Environment, Irrigation = tuple(ZambeziProblem.evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation


# specify model
model = Model('zambeziproblem', function=model_wrapper)
print('after model definition')

# levers
model.levers = [RealParameter('v' + str(i), -1, 1) for i in range(ZambeziProblem.Nvar)]
print('after model.levers')

# specify outcomes
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE),  # Minimize, because deficits
                  ScalarOutcome('Environment', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Irrigation', ScalarOutcome.MINIMIZE)]

print('after model outcomes')

if __name__ == '__main__':
    print('within main statement')
    ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
    ema_logging.log_to_stderr(ema_logging.INFO)

    # with MultiprocessingEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))

    # with SequentialEvaluator(model) as evaluator:
    #     results = evaluator.optimize(nfe=250, searchover="levers", epsilons=[0.1] * len(model.outcomes))
    print('after ema logging')

    # Specify the nfe and add a comment for the run save name
    nfe_n = 1  # if this is 1, the actual number of nfe's is 2.
    run_comment = 'borg_test'  # add a comment to recognize the run output
    seeds_n = 2  # if this is 1, the actual number of seeds is 2

    results = []
    convergences = []

    with MultiprocessingEvaluator(model) as evaluator:
        for i in range(seeds_n):
            # we create 2 convergence tracker metrics
            # the archive logger writes the archive to disk for every x nfe
            # the epsilon progress tracks during runtime
            convergence_metrics = [
                ArchiveLogger(
                    "../data/archives",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes],
                    base_filename=f"{i}.tar.gz",
                )
                , EpsilonProgress(),
            ]

            result, convergence = evaluator.optimize(
                algorithm=GenerationalBorg,
                nfe=int(nfe_n),  # 500000 #250
                searchover="levers",
                epsilons=[0.8] * len(model.outcomes),  # 0.05, 0,1
                convergence=convergence_metrics
            )

            print("result type", type(result))
            print("result", result)

            results.append(result)
            convergences.append(convergence)

            print("results type", type(results))
        print(" results", results)

    print("after evaluator")

    #############
    # CONVERGENCE #TODO: move to notebook
    #############

    all_archives = []

    for i in range(seeds_n):
        archives = ArchiveLogger.load_archives(f"../data/archives/{i}.tar.gz")
        all_archives.append(archives)

    problem = to_problem(model, searchover="levers")

    reference_set = epsilon_nondominated(results, [0.05] * len(model.outcomes), problem)

    hv = HypervolumeMetric(reference_set, problem)
    # gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    # ei = EpsilonIndicatorMetric(reference_set, problem)
    # ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    # sm = SpacingMetric(problem)

    metrics_by_seed = []
    for archives in all_archives:
        metrics = []
        for nfe, archive in archives.items():
            scores = {
                # "generational_distance": gd.calculate(archive),
                "hypervolume": hv.calculate(archive),
                # "epsilon_indicator": ei.calculate(archive),
                # "inverted_gd": ig.calculate(archive),
                # "spacing": sm.calculate(archive),
                "nfe": int(nfe),
            }
            metrics.append(scores)
        metrics = pd.DataFrame.from_dict(metrics)

        # sort metrics by number of function evaluations
        metrics.sort_values(by="nfe", inplace=True)
        metrics_by_seed.append(metrics)

    # Visualize convergence metrics
    sns.set_style("white")
    # fig, axes = plt.plot(nrows=6, figsize=(8, 12), sharex=True)

    # ax1, ax2, ax3, ax4, ax5, ax6 = axes

    for metrics, convergence in zip(metrics_by_seed, convergences):
        ax, fig = plt.plot(metrics.nfe, metrics.hypervolume, label="hypervolume")
        # ax1.set_ylabel("hypervolume")

        '''
        ax2.plot(convergence.nfe, convergence.epsilon_progress)
        ax2.set_ylabel("$\epsilon$ progress")

        ax3.plot(metrics.nfe, metrics.generational_distance)
        ax3.set_ylabel("generational distance")

        ax4.plot(metrics.nfe, metrics.epsilon_indicator)
        ax4.set_ylabel("epsilon indicator")

        ax5.plot(metrics.nfe, metrics.inverted_gd)
        ax5.set_ylabel("inverted generational\ndistance")

        ax6.plot(metrics.nfe, metrics.spacing)
        ax6.set_ylabel("spacing")
        '''

    # ax6.set_xlabel("nfe")

    sns.despine(fig)

    plt.show()

    #####################
    # MERGE ACROSs SEEDS
    #####################

    # Merge the 5 runs of the optimization
    # problem = to_problem(model, searchover="levers")
    epsilons = [0.05] * len(model.outcomes)
    merged_results = epsilon_nondominated(results, epsilons, problem)

    print('merged_results', merged_results)

    # Save the results
    os.chdir('../optimization_results')
    file_name = f"{str(nfe_n)}_{run_comment}"
    print("file name is", file_name)

    # file_path = os.path.join(file_dir, csv_folder, 'findorb_data.txt')
    # dataframe.to_csv(file_path, header=False, index=False)

    # save the results
    # writer = r"../optimization_results"
    #
    merged_results.to_csv(f"merged_results_{file_name}.csv")
