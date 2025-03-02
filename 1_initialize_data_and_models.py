import os

# trunk-ignore(bandit/B403)
import pickle

import numpy as np
from sklearn.model_selection import KFold

from LiCE.data.DataHandler import DataHandler
from LiCE.spn.SPN import SPN
from nn_model import NNModel


def obtain_definition(data_name: str):
    if data_name == "adult":
        from data_config.adult import config
    elif data_name == "gmsc":
        from data_config.gmsc import config
    elif data_name == "credit":
        from data_config.credit import config
    else:
        raise ValueError(f"unsupported data {data_name}")
    return (
        config["X"],
        config["y"],
        config["bounds"],
        config["categ_map"],
        config["ordered"],
        config["discrete"],
        config["causal_inc"],
        config["immutable"],
        config["monotonicity"],
        config["greater_than"],
    )


name = "reproducible_results"

data_names = ["adult", "gmsc", "credit"]
for data_name in data_names:
    (
        X,
        y,
        bounds,
        categ_map,
        ordered,
        discrete,
        causal_inc,
        immutable,
        monotonicity,
        greater_than,
    ) = obtain_definition(data_name)

    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        model_path = f"results/{name}/{data_name}/{i}/models"
        data_path = f"results/{name}/{data_name}/{i}/data"
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        X.iloc[train_index].to_csv(f"{data_path}/X_train.csv")
        X.iloc[test_index].to_csv(f"{data_path}/X_test.csv")
        np.random.seed(0)
        indices = np.random.choice(
            np.arange(test_index.shape[0]), size=(100,), replace=False
        )
        X.iloc[test_index[indices]].to_csv(f"{data_path}/X_subtest.csv")

        y.iloc[train_index].to_csv(f"{data_path}/y_train.csv")
        y.iloc[test_index].to_csv(f"{data_path}/y_test.csv")
        y.iloc[test_index[indices]].to_csv(f"{data_path}/y_subtest.csv")

        dhandler = DataHandler(
            X.iloc[train_index],
            y.iloc[train_index],
            categ_map,
            ordered,
            bounds_map=bounds,
            discrete=discrete,
            immutable=immutable,
            monotonicity=monotonicity,
            causal_inc=causal_inc,
            greater_than=greater_than,
        )
        X_enc = dhandler.encode(X, normalize=True, one_hot=True)
        y_enc = dhandler.encode_y(y, one_hot=False)

        # train nn
        nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
        nn.train(X_enc[train_index], y_enc[train_index])

        nn.save(model_path + "/nn.pt")
        nn.save_onnx(model_path + "/nn.onnx")

        # train spn
        spn = SPN(
            np.concatenate([X.values, y.values], axis=1)[train_index],
            dhandler,
            normalize_data=False,
        )
        with open(model_path + "/spn.pickle", "wb") as f:
            pickle.dump(spn, f)

        with open(model_path + "/dhandler.pickle", "wb") as f:
            pickle.dump(dhandler, f)

        print(f"Done {data_name} - {i}")
