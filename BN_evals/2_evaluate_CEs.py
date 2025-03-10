import pickle

import bnlearn
import numpy as np
import pandas as pd

if __package__ is None:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LiCE.data.DataHandler import DataHandler
from nn_model import NNModel

# trunk-ignore-all(bandit/B301)

# CEs must first be genereated...

data_names = ["asia", "alarm", "win95pts"]
methods = [
    "VAE",
    "DiCE",
    "LiCE_optimize",
    "LiCE_median",
    "MIO_no_spn",
    "CVAE",
    "FACE_knn",
    "FACE_eps",
    "PROPLACE",
]
folds = [0, 1, 2, 3, 4]

path_base = "results/bayes_nets"


def compute_p(CPD, d):
    p = 1
    for c in CPD.keys():
        mask = np.ones(CPD[c].shape[0], dtype=bool)
        for k in CPD[c].columns:
            if k == "p":
                continue
            mask &= CPD[c][k] == d[k]
        p *= CPD[c][mask]["p"].iloc[0]
    return p


def get_info(
    ce,
    ce_target,
    dhandler,
    ll,
    factual_enc,
    factual_enc_num,
    time,
    CPD,
):
    MADs = np.concatenate([f.MAD for f in dhandler.features])
    MADs[MADs < 1e-6] = 1e-6
    invMADs = 1.0 / MADs

    ce_valid_enc = dhandler.encode(ce, normalize=True, one_hot=True)
    ce_valid_enc_num = dhandler.encode(ce, normalize=True, one_hot=False)

    ce_df = pd.DataFrame(ce.reshape(1, -1), columns=dhandler.feature_names)
    ce_full_df = pd.Series(ce.flatten(), index=dhandler.feature_names)
    if target_name == "BP":  # alarm dataset
        if ce_target:
            ce_full_df[target_name] = 2
            bn_prob = compute_p(CPD, ce_full_df)
        else:
            ce_full_df[target_name] = 0
            bn_prob = compute_p(CPD, ce_full_df)
            ce_full_df[target_name] = 1
            bn_prob += compute_p(CPD, ce_full_df)
    else:
        ce_full_df[target_name] = ce_target
        bn_prob = compute_p(CPD, ce_full_df)

    return {
        "CE": ce_df,
        "ll": ll,
        "sparsity": dhandler.n_features
        - np.sum(np.isclose(ce_valid_enc_num, factual_enc_num, atol=1e-6)),
        "distance": np.abs(factual_enc - ce_valid_enc) @ invMADs,
        "time": time,
        "bn_prob": bn_prob,
    }


CEs = {}
for data_name in data_names:
    CEs[data_name] = {m: {} for m in methods}
    model_name = data_name
    if data_name == "asia":
        target_name = "dysp"
    elif data_name == "alarm":
        target_name = "BP"
        target_val = 2
    elif data_name == "win95pts":
        model_name = f"data/BayesianNetworks/{model_name}.bif"
        target_name = "Problem1"

    # load bn
    model = bnlearn.import_DAG(model_name)
    CPD = bnlearn.print_CPD(model, verbose=0)

    for fold in folds:
        print(f"Handling {data_name} - {fold}")
        model_path = f"{path_base}/{data_name}/{fold}/models/"
        data_path = f"{path_base}/{data_name}/{fold}/data/"
        # load the DataHandler
        with open(model_path + "dhandler.pickle", "rb") as f:
            dhandler: DataHandler = pickle.load(f)

        # get data
        X_test = pd.read_csv(f"{data_path}/X_subtest.csv", index_col=0)
        y_test = pd.read_csv(f"{data_path}/y_subtest.csv", index_col=0)
        X_train = pd.read_csv(f"{data_path}/X_train.csv", index_col=0)
        y_train = dhandler.encode_y(
            pd.read_csv(f"{data_path}/y_train.csv", index_col=0)
        )
        X_train_enc = dhandler.encode(X_train, normalize=True, one_hot=True)
        X_train_num = dhandler.encode(X_train, normalize=True, one_hot=False)

        # load spn
        with open(model_path + "spn.pickle", "rb") as f:
            spn = pickle.load(f)

        # load NN
        nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
        nn.load(model_path + "nn.pt")

        # load and filter CEs
        for method in methods:
            results_path = f"{path_base}/{data_name}/{fold}/CEs/{method}.pickle"
            with open(results_path, "rb") as f:
                results = pickle.load(f)

            for i in results.keys():
                id = f"{fold}_{i}"
                CEs[data_name][method][id] = {"time": results[i]["time"]}
                if "stats" in results[i]:
                    CEs[data_name][method][id]["stats"] = results[i]["stats"]

                ces = np.asarray(results[i]["CE"])
                if ces.shape[0] == 0:
                    continue
                if method in ["VAE", "DiCE"]:
                    ces = ces[:, :-1]  # drop the classification

                factual = X_test.loc[i].values
                factual_enc = dhandler.encode(factual)
                factual_enc_num = dhandler.encode(factual, one_hot=False)
                fact_pred = int(nn.predict(factual_enc) > 0)

                # trunk-ignore(ruff/E711)
                none_ces = ces == None
                factual_block = np.repeat(factual.reshape(1, -1), len(ces), axis=0)
                ces[none_ces] = factual_block[none_ces]
                ces_enc = dhandler.encode(ces)
                predictions = (nn.predict(ces_enc) > 0).astype(int).flatten()
                valid = predictions != fact_pred
                if not np.any(valid):
                    continue

                predicitions_decoded = dhandler.decode_y(
                    predictions[valid]
                ).values.reshape(-1, 1)
                ces_valid = ces[valid]
                lls = spn.compute_ll(
                    np.concatenate([ces_valid, predicitions_decoded], axis=1)
                ).flatten()

                max_i = np.argmax(lls)

                CEs[data_name][method][id]["valid"] = get_info(
                    ces_valid[max_i],
                    predicitions_decoded[max_i][0],
                    dhandler,
                    lls[max_i],
                    factual_enc,
                    factual_enc_num,
                    results[i]["time"],
                    CPD,
                )
                if (
                    "stats" in results[i]
                    and len(results[i]["stats"]["ll_computed"]) > 0
                ):
                    ll = np.array(results[i]["stats"]["ll_computed"])[valid][max_i]
                    CEs[data_name][method][id]["valid"]["comp_ll"] = ll
                if ("MIO" in method or "LiCE" in method) and "is_opt" in results[i]:
                    CEs[data_name][method][i]["valid"]["is_opt"] = results[i]["is_opt"]

                ce_actionable = None
                ce_actionable_ll = -np.inf
                for j, (ce, ll) in enumerate(zip(ces_valid, lls)):
                    can_act = True
                    for f, pre, pos in zip(dhandler.features, factual, ce):
                        if not f.allowed_change(pre, pos, encoded=False):
                            can_act = False
                            break
                    if can_act:
                        if ce_actionable_ll < ll:
                            ce_actionable_ll = ll
                            ce_actionable = ce
                            act_i = j

                if ce_actionable is not None:
                    CEs[data_name][method][id]["actionable"] = get_info(
                        ce_actionable,
                        predicitions_decoded[max_i][0],
                        dhandler,
                        ce_actionable_ll,
                        factual_enc,
                        factual_enc_num,
                        results[i]["time"],
                        CPD,
                    )
                    if (
                        "stats" in results[i]
                        and len(results[i]["stats"]["ll_computed"]) > 0
                    ):
                        ll = np.array(results[i]["stats"]["ll_computed"])[valid][act_i]
                        CEs[data_name][method][id]["valid"]["comp_ll"] = ll

with open(path_base + "/results.pickle", "wb") as f:
    pickle.dump(CEs, f)
