import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import os
from scipy import stats

from matplotlib import pyplot
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
import torch
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer


def dump_artifact(artifact, name):
  pickle_file = open(DATA_DIR + f'artifacts/{name}', 'wb')
  pickle.dump(artifact, pickle_file)
  pickle_file.close()

def load_artifact(name):
  pickle_file = open(DATA_DIR + f'artifacts/{name}', 'rb')
  artifact = pickle.load(pickle_file)
  pickle_file.close()
  return artifact

def membership_advantage(y_true, y_pred, sample_weight):
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    ma = tpr - fpr
    ma = (ma + 1) / 2
    return ma

def activate_3(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities


DATA_DIR = "/Users/golobs/Documents/GradSchool/SNAKE/"
name_of_attack = "BNAF_DOMIAS_ordinalFeaturesOnly"


verbose = False
use_solutions = True

data = "main"
n_runs = 20

if not os.path.exists(DATA_DIR + f'{name_of_attack}'):
  os.mkdir(DATA_DIR + f'{name_of_attack}')

# for gen in ['mst', 'pategan', 'privbayes']:
#   for eps in [1, 10, 100, 1000]:

for gen in ['pategan']:
  for eps in [100]:

    targets = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_targets.csv")
    targets_idx = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_targets_index.txt", names=['hhid'])
    Y_test = pd.read_csv(DATA_DIR + f"{data}_solutions/{gen}_{eps}.txt", names=['membership'])['membership'] if use_solutions else [0]*targets_idx.shape[0]
    targets_idx['actual'] = Y_test
    targets_hhid = targets['hhid']

    targets_encoded = load_artifact(f"{data}_{gen}_{eps}_BNAF_targets_encoded")
    target_results_synth = np.array([0.0] * targets.shape[0])
    target_results_base = np.array([0.0] * targets.shape[0])

    for i in range(n_runs):
        p_G_model = load_artifact(f"{data}_{gen}_{eps}_synth_BNAF_model_i{i}")
        p_R_model = load_artifact(f"{data}_{gen}_{eps}_base_BNAF_model_i{i}")

        p_G_evaluated = np.exp(
            compute_log_p_x(p_G_model, torch.as_tensor(targets_encoded).float().to('cpu'))
            .cpu()
            .detach()
            .numpy()
        )

        p_R_evaluated = np.exp(
            compute_log_p_x(p_R_model, torch.as_tensor(targets_encoded).float().to('cpu'))
            .cpu()
            .detach()
            .numpy()
        )

        p_R_evaluated = np.nan_to_num(p_R_evaluated, nan=1e26)
        p_G_evaluated = np.nan_to_num(p_G_evaluated, nan=1e26)

        p_R_evaluated[p_R_evaluated == 0] = 1e26
        p_G_evaluated[p_G_evaluated == 0] = 1e26

        # # density_gen = stats.gaussian_kde(synth_set.transpose(1, 0))
        # # density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
        # # p_G_evaluated = density_gen(X_test.transpose(1, 0))
        # # p_R_evaluated = density_data(X_test.transpose(1, 0))


        target_results_synth += p_R_evaluated
        target_results_base += p_G_evaluated

        p_rel = p_R_evaluated / p_G_evaluated

        scores = pd.DataFrame({
            "hhid": targets_hhid,
            "p_rel": pd.Series(p_rel)
        })

        # scores["prob"] = activate_3(scores["p_rel"])

        final_scores = []
        households = scores.groupby("hhid")
        for target_hhid in targets_idx['hhid'].values.tolist():
            final_scores.append(households.get_group(target_hhid).p_rel.mean())

        final_scores = activate_3(np.array(final_scores), confidence=1)
        # np.savetxt(DATA_DIR + f'{name_of_attack}/{gen}_{eps}.txt', final_scores, fmt="%.8f")

        if verbose:
            bins = np.linspace(0, 1, 50)
            pyplot.hist(final_scores, bins)
            pyplot.legend(loc='upper right')
            pyplot.title(f"epsilon ~ {eps}")
            pyplot.show()

        if use_solutions:
            weights = 2 * np.abs(0.5 - final_scores)
            ma = membership_advantage(Y_test, final_scores > 0.5, sample_weight=weights)
            print(i, ma)

        if verbose: print()
        if verbose: print(pd.Series(final_scores).describe())


    ## AVERAGE MANY RUNS
    p_rel = target_results_synth / target_results_base

    scores = pd.DataFrame({
        "hhid": targets_hhid,
        "p_rel": pd.Series(p_rel)
    })

    # scores["prob"] = activate_3(scores["p_rel"])

    final_scores = []
    households = scores.groupby("hhid")
    for i, target_hhid in enumerate(targets_idx['hhid'].values.tolist()):
      final_scores.append(households.get_group(target_hhid).p_rel.mean())

    final_scores = activate_3(np.array(final_scores), confidence=1)
    # np.savetxt(DATA_DIR + f'{name_of_attack}/{gen}_{eps}.txt', final_scores, fmt="%.8f")

    if verbose:
      bins = np.linspace(0, 1, 50)
      pyplot.hist(final_scores, bins)
      pyplot.legend(loc='upper right')
      pyplot.title(f"epsilon ~ {eps}")
      pyplot.show()

    if use_solutions:
        weights = 2 * np.abs(0.5 - final_scores)
        ma = membership_advantage(Y_test, final_scores > 0.5, sample_weight=weights)
        print(ma)

    if verbose: print()
    if verbose: print(pd.Series(final_scores).describe())
