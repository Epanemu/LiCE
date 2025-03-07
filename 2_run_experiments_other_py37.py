# trunk-ignore-all(bandit/B403)
import copy
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch.nn as tnn
from carla.recourse_methods import CCHVAE, Face

import compared_methods.proplace.proplace as prop
from compared_methods.carla_helpers import MixedPolytopeDataset, MyNNModel
from compared_methods.proplace.experiments.exputils import (
    get_test_data,
    get_test_data_1,
    run_proplace,
    run_rnnce,
    run_rnnce_class0,
)
from LiCE.data.Features import Categorical, Contiguous
from nn_model import NNModel

# trunk-ignore-all(bandit/B301)

data_names = [sys.argv[1]]
folds = [int(sys.argv[2])]
proplace_run = sys.argv[3] == "proplace"
folder = sys.argv[4]

prefix = f"results/{folder}"

print(data_names, folds, proplace_run)


for fold in folds:
    for data_name in data_names:
        path_base = f"{prefix}/{data_name}/{fold}"

        with open(f"{path_base}/models/dhandler.pickle", "rb") as f:
            dhandler = pickle.load(f)

        nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
        nn.load(f"{path_base}/models/nn.pt")

        X_test = pd.read_csv(f"{path_base}/data/X_subtest.csv", index_col=0)
        y_test = pd.read_csv(f"{path_base}/data/y_subtest.csv", index_col=0)

        X_train = pd.read_csv(f"{path_base}/data/X_train.csv", index_col=0)
        y_train = pd.read_csv(f"{path_base}/data/y_train.csv", index_col=0)

        # PREPARATIONS

        nn_sigmoid = copy.deepcopy(nn)
        nn_sigmoid.model.append(tnn.Sigmoid())

        # PROPLACE
        if proplace_run:
            continuous = [
                f.name for f in dhandler.features if isinstance(f, Contiguous)
            ]
            one_hot_cols = []
            immutable = []
            for f in dhandler.features:
                if isinstance(f, Categorical):
                    for val in f.orig_vals:
                        one_hot_cols.append(f"{f.name}_{val}")
                        if not f.modifiable:
                            immutable.append(f"{f.name}_{val}")
                else:
                    one_hot_cols.append(f.name)
                    if not f.modifiable:
                        immutable.append(f.name)

            X_train_enc = pd.DataFrame(
                dhandler.encode(X_train, one_hot=True), columns=one_hot_cols
            )
            y_train_enc = pd.DataFrame(
                dhandler.encode_y(y_train, one_hot=False),
                columns=[dhandler.target_feature.name],
            )
            X_test_enc = pd.DataFrame(
                dhandler.encode(X_test, one_hot=True), columns=one_hot_cols
            )
            y_test_enc = pd.DataFrame(
                dhandler.encode_y(y_test, one_hot=False),
                columns=[dhandler.target_feature.name],
            )
            dataset = prop.InnDataSet(
                X_train_enc, y_train_enc, X_test_enc, y_test_enc, continuous, immutable
            )

            nn_proplace = prop.clfutils.InnModel(dataset, nn_sigmoid.model, [20, 10])

            # epochs = 50
            epochs = 15
            m2s = prop.retrain_models(
                dataset, [20, 10], epochs
            ) + prop.retrain_models_leave_some_out(dataset, [20, 10], epochs)
            clf = nn_proplace
            test_xs_vals, test_xs, test_xs_carla, utildataset, nodes = get_test_data(
                nn_proplace, m2s, dataset
            )
            test_xs_vals_1, _, _, utildataset_1, nodes_1 = get_test_data_1(
                nn_proplace, m2s, dataset
            )
            delta_1 = 0.025
            try:
                rnnce_ces, treer, X_class1_clf_robust = run_rnnce(
                    dataset, clf, nodes, utildataset, delta_1, test_xs_vals
                )
            except ValueError:
                X_class1_clf_robust = np.array([[1, 2]])
            while X_class1_clf_robust.shape[0] < X_class1_clf_robust.shape[1]:
                delta_1 -= 0.005
                try:
                    rnnce_ces, treer, X_class1_clf_robust = run_rnnce(
                        dataset, clf, nodes, utildataset, delta_1, test_xs_vals
                    )
                except ValueError:
                    X_class1_clf_robust = np.array([[1, 2]])

            # extension for the other class
            delta_0 = 0.025
            try:
                rnnce_ces_0, treer_0, X_class0_clf_robust = run_rnnce_class0(
                    dataset, clf, nodes_1, utildataset_1, delta_0, test_xs_vals_1
                )
            except ValueError:
                X_class0_clf_robust = np.array([[1, 2]])
            while X_class0_clf_robust.shape[0] < X_class0_clf_robust.shape[1]:
                delta_0 -= 0.005
                try:
                    rnnce_ces_0, treer_0, X_class0_clf_robust = run_rnnce_class0(
                        dataset, clf, nodes_1, utildataset_1, delta_0, test_xs_vals_1
                    )
                except ValueError:
                    X_class0_clf_robust = np.array([[1, 2]])

        # CARLA LIB
        else:
            one_hot_cols = []
            immutable = []
            for f in dhandler.features:
                if isinstance(f, Categorical):
                    for val in f.orig_vals:
                        one_hot_cols.append(f"{f.name}_{val}")
                        if not f.modifiable:
                            immutable.append(f"{f.name}_{val}")
                else:
                    one_hot_cols.append(f.name)
                    if not f.modifiable:
                        immutable.append(f.name)

            data_train = pd.concat([X_train, y_train], axis=1)
            data_test = pd.concat([X_test, y_test], axis=1)
            carla_dataset = MixedPolytopeDataset(
                dhandler, data_train, data_test, one_hot_cols
            )

            carla_nn = MyNNModel(carla_dataset, nn)

            hyperparameters = {"data_name": data_name, "mode": "knn"}
            if X_train.shape[0] <= 1000:
                hyperparameters["fraction"] = 0.5
            face_knn = Face(mlmodel=carla_nn, hyperparams=hyperparameters)

            hyperparameters = {"data_name": data_name, "mode": "epsilon"}
            if X_train.shape[0] <= 1000:
                hyperparameters["fraction"] = 0.5
            face_eps = Face(mlmodel=carla_nn, hyperparams=hyperparameters)

            w = sum(f.encoding_width(True) for f in dhandler.features if f.modifiable)

            hyperparameters = {
                "data_name": data_name,
                "vae_params": {"layers": [w, 20, 10]},
            }
            if X_train.shape[0] <= 1000:
                hyperparameters["vae_params"]["epochs"] = 50
            cchvae = CCHVAE(mlmodel=carla_nn, hyperparams=hyperparameters)

        results_face_eps = {}
        results_face_knn = {}
        results_cvae = {}
        results_proplace = {}
        for i, _ in X_test.iterrows():
            enc_sample = dhandler.encode(X_test.loc[[i]])
            prediction = nn.predict(enc_sample) > 0

            if proplace_run:
                t = time.process_time()
                if prediction:
                    try:
                        proplace_res = run_proplace(
                            clf,
                            nodes_1,
                            utildataset_1,
                            delta_0,
                            treer_0,
                            X_class0_clf_robust,
                            enc_sample,
                            k=dhandler.encoding_width(True),
                            yprime=0,
                        )
                        print("success")
                    except AttributeError:
                        proplace_res = []
                else:
                    try:
                        proplace_res = run_proplace(
                            clf,
                            nodes,
                            utildataset,
                            delta_1,
                            treer,
                            X_class1_clf_robust,
                            enc_sample,
                            k=dhandler.encoding_width(True),
                            yprime=1,
                        )
                        print("success")
                    except AttributeError:
                        proplace_res = []
                tdiff = time.process_time() - t
                results_proplace[i] = {
                    "CE": dhandler.decode(
                        np.asarray(proplace_res).round(8),
                        denormalize=True,
                        encoded_one_hot=True,
                        as_dataframe=False,
                    ),
                    "ll_threshold": None,
                    "is_opt": False,
                    "time": tdiff,
                    "deltas": (delta_0, delta_1),
                }

            else:
                pdenc_sample = pd.DataFrame(enc_sample, columns=one_hot_cols)

                t = time.process_time()
                face_eps_res = face_eps.get_counterfactuals(pdenc_sample)
                tdiff = time.process_time() - t
                if np.any(np.isnan(face_eps_res)):
                    print("got nans", "face_eps")
                results_face_eps[i] = {
                    "CE": (
                        []
                        if np.any(np.isnan(face_eps_res))
                        else dhandler.decode(face_eps_res.values, as_dataframe=False)
                    ),
                    "ll_threshold": None,
                    "is_opt": False,
                    "time": tdiff,
                }

                t = time.process_time()
                face_knn_res = face_knn.get_counterfactuals(pdenc_sample)
                tdiff = time.process_time() - t
                if np.any(np.isnan(face_knn_res)):
                    print("got nans", "face_knn")
                results_face_knn[i] = {
                    "CE": (
                        []
                        if np.any(np.isnan(face_knn_res))
                        else dhandler.decode(face_knn_res.values, as_dataframe=False)
                    ),
                    "ll_threshold": None,
                    "is_opt": False,
                    "time": tdiff,
                }

                t = time.process_time()
                cvae_res = cchvae.get_counterfactuals(pdenc_sample)
                tdiff = time.process_time() - t
                if np.any(np.isnan(cvae_res)):
                    print("got nans", "cvae")
                results_cvae[i] = {
                    "CE": (
                        []
                        if np.any(np.isnan(cvae_res))
                        else dhandler.decode(cvae_res.values, as_dataframe=False)
                    ),
                    "ll_threshold": None,
                    "is_opt": False,
                    "time": tdiff,
                }

        os.makedirs(f"{path_base}/CEs", exist_ok=True)

        if proplace_run:
            with open(f"{path_base}/CEs/PROPLACE.pickle", "wb") as f:
                pickle.dump(results_proplace, f)

        else:
            with open(f"{path_base}/CEs/FACE_knn.pickle", "wb") as f:
                pickle.dump(results_face_knn, f)

            with open(f"{path_base}/CEs/FACE_eps.pickle", "wb") as f:
                pickle.dump(results_face_eps, f)

            with open(f"{path_base}/CEs/CVAE.pickle", "wb") as f:
                pickle.dump(results_cvae, f)
