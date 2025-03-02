# trunk-ignore-all(bandit/B403)
import pickle

import numpy as np
import pandas as pd

from LiCE.data.DataHandler import DataHandler
from nn_model import NNModel

# trunk-ignore-all(bandit/B301)

data_names = ["credit", "adult", "gmsc"]
methods = [
    "VAE",
    "DiCE",
    "LiCE_optimize",
    "LiCE_median",
    "LiCE_quartile",
    "LiCE_sample",
    "MIO_no_spn",
    "CVAE",
    "FACE_knn",
    "FACE_eps",
    "PROPLACE",
]
folds = [0, 1, 2, 3, 4]

path_base = "results/reproducible_results"


def get_info(ce, dhandler, factual_enc, factual_enc_num, time):
    MADs = np.concatenate([f.MAD for f in dhandler.features])
    MADs[MADs < 1e-6] = 1e-6
    invMADs = 1.0 / MADs

    ce_valid_enc = dhandler.encode(ce, normalize=True, one_hot=True)
    ce_valid_enc_num = dhandler.encode(ce, normalize=True, one_hot=False)

    return {
        "CE": pd.DataFrame(ce.reshape(1, -1), columns=dhandler.feature_names),
        "ll": lls[max_i],
        "sparsity": dhandler.n_features
        - np.sum(np.isclose(ce_valid_enc_num, factual_enc_num, atol=1e-6)),
        "distance": np.abs(factual_enc - ce_valid_enc) @ invMADs,
        "time": time,
    }


CEs = {}
for data_name in data_names:
    CEs[data_name] = {m: {} for m in methods}
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
                CEs[data_name][method][i] = {"time": results[i]["time"]}
                if "stats" in results[i]:
                    CEs[data_name][method][i]["stats"] = results[i]["stats"]

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

                CEs[data_name][method][i]["valid"] = get_info(
                    ces_valid[max_i],
                    dhandler,
                    factual_enc,
                    factual_enc_num,
                    results[i]["time"],
                )
                if (
                    "stats" in results[i]
                    and len(results[i]["stats"]["ll_computed"]) > 0
                ):
                    ll = np.array(results[i]["stats"]["ll_computed"])[valid][max_i]
                    CEs[data_name][method][i]["valid"]["comp_ll"] = ll
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
                    CEs[data_name][method][i]["actionable"] = get_info(
                        ce_actionable,
                        dhandler,
                        factual_enc,
                        factual_enc_num,
                        results[i]["time"],
                    )
                    if (
                        "stats" in results[i]
                        and len(results[i]["stats"]["ll_computed"]) > 0
                    ):
                        ll = np.array(results[i]["stats"]["ll_computed"])[valid][act_i]
                        CEs[data_name][method][i]["actionable"]["comp_ll"] = ll

with open(path_base + "/results.pickle", "wb") as f:
    pickle.dump(CEs, f)
