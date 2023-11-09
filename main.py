# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import json
import sys
import numpy as np
import pickle
import stats


sys.path.append('reprosyn-main/src/reprosyn/methods/gans/')

import disjoint_set
# import mst
# import privbayes
import gans

DATA_DIR = "/Users/golobs/Documents/GradSchool/SNAKE/"

def dump_artifact(artifact, name):
    pickle_file = open(DATA_DIR + f'{name}', 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()



# TODO: fix encoding adjustment

n_runs = 20
data = "playground"
gen = "pategan"

with open(DATA_DIR + "meta.json") as f:
    meta2 = json.load(f)

base = pd.read_parquet(DATA_DIR + "base.parquet")
base['HHID'] = base.index
base.index = range(base.shape[0])
columns = base[np.take(base.columns, range(15))].columns

for eps in [1, 10, 100, 1000]:
    print(eps)
    synth = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_synthetic.csv")
    targets = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_targets.csv")

    synth_scores = np.zeros(targets.shape[0])
    base_scores = np.zeros(targets.shape[0])
    for j in range(n_runs):
        print(j, end=" ")
        gen_synth = gans.PATEGAN(
            dataset=synth[columns],
            targets=targets[columns],
            metadata=meta2,
            size=10000,
            epsilon=100_000_000,
            # epsilon=eps,
        )
        gen_synth.run()
        synth_scores += gen_synth.target_results

        gen_base = gans.PATEGAN(
            # dataset=base[columns],
            dataset=pd.concat([base[columns].sample(n=10000), targets[columns]], ignore_index=True),
            targets=targets[columns],
            metadata=meta2,
            size=10000,
            epsilon=100_000_000,
            # epsilon=eps,
        )
        gen_base.run()
        base_scores += gen_base.target_results

    synth_scores /= n_runs
    base_scores /= n_runs

    np.savetxt(DATA_DIR + f'artifacts/r7_{data}_{gen}_{eps}.txt', synth_scores / base_scores, fmt="%.6f")
    # np.savetxt(DATA_DIR + f'artifacts/s7_{data}_{gen}_{eps}.txt', synth_scores, fmt="%.6f")
    print()


# r3 = 100 runs
# r4 = no noise


