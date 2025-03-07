import os
import pickle

if __package__ is None:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import bnlearn
import numpy as np

from LiCE.data.DataHandler import DataHandler
from LiCE.spn.SPN import SPN
from nn_model import NNModel


def obtain_definition(data_name: str, seeds: int = 5):
    n_train = 10000
    n_test = 100
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    target_val = None

    model_name = data_name
    if data_name == "asia":
        target = "dysp"
    elif data_name == "alarm":
        target = "BP"
        target_val = 2
    elif data_name == "win95pts":
        model_name = f"data/BayesianNetworks/{model_name}.bif"
        target = "Problem1"
    else:
        raise ValueError(f"unsupported data {data_name}")

    model = bnlearn.import_DAG(model_name)
    CPD = bnlearn.print_CPD(model, verbose=0)
    domains = {}
    for k, df in CPD.items():
        domains[k] = df[k].unique()
    for seed in range(seeds):
        np.random.seed(seed)
        train = bnlearn.sampling(model, n_train, verbose=False)
        test = bnlearn.sampling(model, n_test, verbose=False)

        train_data.append(train[[c for c in train.columns if c != target]])
        train_target.append(
            train[[target]] if target_val is None else train[[target]] == target_val
        )
        test_data.append(test[[c for c in test.columns if c != target]])
        test_target.append(
            test[[target]] if target_val is None else test[[target]] == target_val
        )

    return (
        train_data,
        test_data,
        train_target,
        test_target,
        domains,
    )


if __name__ == "__main__":
    name = "bayes_nets"

    data_names = ["asia", "alarm", "win95pts"]

    seeds = 5

    for data_name in data_names:
        (
            Xs,
            Xtests,
            ys,
            ytests,
            categ_map,
        ) = obtain_definition(data_name, seeds=seeds)

        for i in range(seeds):

            model_path = f"results/{name}/{data_name}/{i}/models"
            data_path = f"results/{name}/{data_name}/{i}/data"
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)

            Xs[i].to_csv(f"{data_path}/X_train.csv")
            Xtests[i].to_csv(f"{data_path}/X_test.csv")
            np.random.seed(i)
            Xtests[i].to_csv(f"{data_path}/X_subtest.csv")

            ys[i].to_csv(f"{data_path}/y_train.csv")
            ytests[i].to_csv(f"{data_path}/y_test.csv")
            ytests[i].to_csv(f"{data_path}/y_subtest.csv")

            dhandler = DataHandler(
                Xs[i],
                ys[i],
                categ_map,
            )
            X_enc = dhandler.encode(Xs[i], normalize=True, one_hot=True)
            y_enc = dhandler.encode_y(ys[i], one_hot=False)

            # train nn
            nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
            nn.train(X_enc, y_enc)

            nn.save(model_path + "/nn.pt")
            nn.save_onnx(model_path + "/nn.onnx")

            # train spn
            spn = SPN(
                np.concatenate([Xs[i].values, ys[i].values], axis=1),
                dhandler,
                normalize_data=False,
            )
            with open(model_path + "/spn.pickle", "wb") as f:
                pickle.dump(spn, f)

            with open(model_path + "/dhandler.pickle", "wb") as f:
                pickle.dump(dhandler, f)

            print(f"Done {data_name} - {i}")
