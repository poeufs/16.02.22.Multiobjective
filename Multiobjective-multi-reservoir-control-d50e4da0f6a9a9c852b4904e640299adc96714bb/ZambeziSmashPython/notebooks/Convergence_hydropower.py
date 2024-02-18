import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from ema_workbench import (MultiprocessingEvaluator, ema_logging, RealParameter, ScalarOutcome, Constant,
                           Model, HypervolumeMetric, save_results)
from ema_workbench.em_framework.optimization import (GenerationalBorg, epsilon_nondominated, to_problem, ArchiveLogger,
                                                     EpsilonProgress)

if not os.path.exists("../src"):
    os.chdir("../../src")
else:
    os.chdir('../src')
os.getcwd()

### BASE CASE

from model_zambezi_OPT import ModelZambezi

ZambeziProblem = ModelZambezi()


def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    # print('len kwargs is', len(kwargs)) = 230
    Hydropower, Environment, Irrigation = tuple(ZambeziProblem.evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation


# specify model
model = Model('zambeziproblem', function=model_wrapper)

# levers
model.levers = [RealParameter('v' + str(i), -1, 1) for i in range(ZambeziProblem.Nvar)]

# specify outcomes
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE),  # Minimize, because deficits
                  ScalarOutcome('Environment', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Irrigation', ScalarOutcome.MINIMIZE)]

# Problem definition
problem = to_problem(model, searchover="levers")

nfe = 200000  #
seeds = 5
run_comment = 'BC'
run_label = f"{run_comment}_{nfe}nfe_{seeds}seed"

os.chdir(f'../runs/{run_label}')
os.getcwd()


import tarfile
import os
import pandas as pd


def alternative_load_archives(filename):
    archives = {}
    with tarfile.open(os.path.abspath(filename)) as fh:
        for entry in fh.getmembers():
            if entry.name.endswith("csv"):
                key = entry.name.split("/")[1][:-4]
                # print(entry.name)
                df = pd.read_csv(fh.extractfile(entry), index_col=0)
                if not df.empty:
                    archives[int(key)] = df
    # print(archives)
    return archives


ArchiveLogger.load_archives = alternative_load_archives

convergences = []
for i in range(seeds):
    df = pd.read_csv(f"convergence{i}.csv")
    convergences.append(df)
    # f'convergence{i}' = pd.read_csv(f"{run_name}/convergence{i}.csv")
    # print(f'convergence{i}')

# Load the archives
all_archives = []
for i in range(seeds):
    archives = ArchiveLogger.load_archives(f"archives/{i}.csv")
    # archives.items()[nfe] =
    # archives = archives.loc[:, ~archives.columns.str.contains('^Unnamed')]
    all_archives.append(archives)

column_names = ['Hydropower', 'Environment', 'Irrigation']

results_list = []
for i in range(seeds):
    result = pd.read_csv(
        f"results_seed{i}.csv")  # Create the results list, containing dataframes with the results per seed
    globals()[f'df_{i}'] = pd.read_csv(f"results_seed{i}.csv",
                                       usecols=column_names)  # create dataframes per seed with results for analysis
    # archives.items()[nfe] =
    # archives = archives.loc[:, ~archives.columns.str.contains('^Unnamed')]
    results_list.append(result)

# Define the reference list
reference_set = epsilon_nondominated(results_list, [0.4] * len(model.outcomes), problem)  # [0.05]
len(reference_set)
# print('reference_set type is', type(reference_set))

# Define the hypervolumemetric
hv = HypervolumeMetric(reference_set, problem)

from datetime import datetime

before = datetime.now()
print("time before is", before)
'''
# Calculate the metrics
metrics_by_seed = []
for archives in all_archives:
    metrics = []
    for nfe, archive in archives.items():
        scores = {
            "hypervolume": hv.calculate(archive),
            "nfe": int(nfe),
        }
        metrics.append(scores)
    metrics = pd.DataFrame.from_dict(metrics)
    metrics.sort_values(by="nfe", inplace=True)
    metrics_name = 'metrics.csv'
    # metrics.to_csv(os.path.join(cwd, metrics_name))

    metrics_by_seed.append(metrics)

# Visualize convergence metrics
sns.set_style("white")
fig, axes = plt.subplots(nrows=2, figsize=(8, 12), sharex=True)

ax1, ax2 = axes
import matplotlib.pyplot as plt

for metrics, convergence in zip(metrics_by_seed, convergences):
    # plt.rcParams["font.family"] = "sans-serif"
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel("$\epsilon$ progress")

    ax2.plot(metrics.nfe, metrics.hypervolume)
    ax2.set_ylabel("hypervolume")

sns.despine(fig)

plt.show()

after = datetime.now()
print(f"It took {after - before} time to do {nfe} nfes")
'''
import sys

sys.path
path = r'C:\users\whitl\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages'
sys.path.append(path)
sys.path

import hvwfg

#seed_keys = []
#result_values = []
#for key, value in archives.items():
 #   seed_keys.append(key)
 #   for i in range(len(seed_keys)):
# print("Key:", key)#, "Value:", value)

# Fitness vector assuming minimization
obj = np.array([[0.3, 0.6],
                [0.4, 0.4],
                [0.6, 0.2]])

ref = np.array([1.1, 1.1])

hvwfg.wfg(obj, ref)

import tarfile
import os
import pandas as pd


def alternative_load_archives(filename):
    archives = {}
    with tarfile.open(os.path.abspath(filename)) as fh:
        for entry in fh.getmembers():
            if entry.name.endswith("csv"):
                key = entry.name.split("/")[1][:-4]
                # print(entry.name)
                df = pd.read_csv(fh.extractfile(entry), index_col=0)
                if not df.empty:
                    archives[int(key)] = df
    # print(archives)
    return archives


ArchiveLogger.load_archives = alternative_load_archives

# Load the archives
all_archives = []
all_hvs = []
for i in range(seeds):
    hv_this_seed = {}
    archives = ArchiveLogger.load_archives(f"archives/{i}.csv")
    # archives.items()[nfe] =
    # archives = archives.loc[:, ~archives.columns.str.contains('^Unnamed')]
    all_archives.append(archives)
    # for key, value in archives.items():
    # if key < 150:
    # print("Key:", key, "Value:", value)
    if i < 1:
        sorted_archives = dict(sorted(archives.items()))
        # Print the sorted dictionary
        for key, value in sorted_archives.items():
            array_reference_set = reference_set.values.copy()
            o = value.values
            objs = np.ascontiguousarray(o)
            objs_new = objs[230::]
            hv = hvwfg.wfg(objs,
                           array_reference_set)

            print("Key:", hv)  # , "Value:", value)
for i in range(1):
    print(objs_new)
    # for i in range(len(all_archives)):
#    print(i)
# print(all_archives)