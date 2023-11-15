import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import os
from scipy import stats

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

verbose = True
data = "playground"

meta = pd.read_json(DATA_DIR + "meta.json")

base = pd.read_parquet(DATA_DIR + "base.parquet")
base['HHID'] = base.index
base.index = range(base.shape[0])


ordered_columns_idx = [0, 5, 7, 11, 12]
ordered_columns = ['age', 'ownchild', 'gradeatn', 'hoursut', 'faminc']
categorical_columns_idx = [column_idx for column_idx in range(15) if column_idx not in ordered_columns_idx]
categorical_columns = [column_name for column_name in meta.name.values.tolist() if column_name not in ordered_columns]

all_columns = categorical_columns
# all_columns = base.loc[:, base.columns != "HHID"].columns

oh_enc = OneHotEncoder(sparse=False, categories=[meta.representation.values.tolist()[i] for i in categorical_columns_idx])
# oh_enc = OneHotEncoder(sparse=False, categories=[meta.representation.values.tolist()[i] for i in range(15)])
oh_enc.fit(base[all_columns])
base_encoded = pd.DataFrame(oh_enc.transform(base[all_columns]), columns=oh_enc.get_feature_names_out(all_columns))


# for data in ['playground', 'main', 'main2']:
# for gen in ['mst', 'pategan', 'privbayes']:
for gen in ['pategan']:
    for eps in [1, 10, 100, 1000]:
  # for eps in [1000]:
        print(f"GENERATOR: {gen}")
        print(f"EPSILON: {eps}")

        synth = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_synthetic.csv")
        targets = pd.read_csv(DATA_DIR + f"public_data_{data}/{gen}_{eps}_targets.csv")

        #### encode data
        ##---------------------------------------
        synth_encoded = pd.DataFrame(oh_enc.transform(synth[all_columns]), columns=oh_enc.get_feature_names_out(all_columns))
        targets_encoded = pd.DataFrame(oh_enc.transform(targets[all_columns]), columns=oh_enc.get_feature_names_out(all_columns))

        base_encoded_sample = base_encoded.sample(n=10000)._append(targets_encoded).sample(frac=1.0).to_numpy()
        synth_encoded = synth_encoded.to_numpy()
        targets_encoded = targets_encoded.to_numpy()
        dump_artifact(targets_encoded, f"{data}_{gen}_{eps}_BNAF_targets_encoded_allfeatures")

        #### build BNAF model
        ##---------------------------------------
        split = .5

        _, p_G_model = density_estimator_trainer(
            synth_encoded,
            synth_encoded[: int(split * synth.shape[0])],
            synth_encoded[int(split * synth.shape[0]) :],
            epochs=50,
            batch_dim=400,
        )
        dump_artifact(p_G_model, f"{data}_{gen}_{eps}_synth_BNAF_model_allfeatures")

        _, p_R_model = density_estimator_trainer(
            base_encoded_sample,
            base_encoded_sample[: int(split * base_encoded_sample.shape[0])],
            base_encoded_sample[int(split * base_encoded_sample.shape[0]) :],
            epochs=50,
            batch_dim=400,
        )
        dump_artifact(p_R_model, f"{data}_{gen}_{eps}_base_BNAF_model_allfeatures")

