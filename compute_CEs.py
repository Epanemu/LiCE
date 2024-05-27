import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from LiCE.lice.LiCE import LiCE
from nn_model import NNModel

# trunk-ignore-all(bandit/B301)
# trunk-ignore-all(bandit/B403)

prefix = f"results/{sys.argv[1]}"
time_limit = int(sys.argv[2])
data_names = [sys.argv[3]]
folds = [int(sys.argv[4])]
median = sys.argv[5] == "median"
quartile = sys.argv[5] == "quartile"
optimize = sys.argv[5] == "optimize"

print(prefix, time_limit, data_names, folds, median, quartile, optimize)

for data_name in data_names:
    for fold in folds:
        path_base = f"{prefix}/{data_name}/{fold}"
        with open(f"{path_base}/models/spn.pickle", "rb") as f:
            spn = pickle.load(f)
        with open(f"{path_base}/models/dhandler.pickle", "rb") as f:
            dhandler = pickle.load(f)

        nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
        nn.load(f"{path_base}/models/nn.pt")

        X_test = pd.read_csv(f"{path_base}/data/X_subtest.csv", index_col=0)
        y_test = pd.read_csv(f"{path_base}/data/y_subtest.csv", index_col=0)

        X_train = pd.read_csv(f"{path_base}/data/X_train.csv", index_col=0)
        y_train = pd.read_csv(f"{path_base}/data/y_train.csv", index_col=0)
        train_data = np.concatenate([X_train.values, y_train.values], axis=1)
        lls = spn.compute_ll(train_data)
        median_ll = np.median(lls)
        quartile_ll = np.percentile(lls, 0.25)

        lice = LiCE(
            spn,
            nn_path=f"{path_base}/models/nn.onnx",
            data_handler=dhandler,
        )

        results_median = {}
        results_quartile = {}
        results_optimize = {}
        results_sample = {}
        results_nospn = {}
        for i, sample in X_test.iterrows():
            sample_ll = spn.compute_ll(
                np.concatenate([sample.values, y_test.loc[i].values])
            )[0]
            enc_sample = dhandler.encode(X_test.loc[[i]])
            prediction = nn.predict(enc_sample) > 0

            if optimize:
                t = time.process_time()
                opt, optimize_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    optimize_ll=True,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                )
                tdiff = time.process_time() - t
                results_optimize[i] = {
                    "CE": optimize_res,
                    "ll_threshold": None,
                    "is_opt": opt,
                    "time": tdiff,
                }
                print(f"done optimize {opt}, {len(optimize_res)}")
            elif quartile:
                t = time.process_time()
                opt, quartile_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    quartile_ll,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                )
                tdiff = time.process_time() - t
                results_quartile[i] = {
                    "CE": quartile_res,
                    "ll_threshold": quartile_ll,
                    "is_opt": opt,
                    "time": tdiff,
                }
                print(f"done quartile {opt}, {len(quartile_res)}")
            elif median:
                t = time.process_time()
                opt, median_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    median_ll,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                )
                tdiff = time.process_time() - t
                results_median[i] = {
                    "CE": median_res,
                    "ll_threshold": median_ll,
                    "is_opt": opt,
                    "time": tdiff,
                }
                print(f"done median {opt}, {len(median_res)}")

            else:
                t = time.process_time()
                opt, sample_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    min(median_ll, sample_ll),
                    n_counterfactuals=10,
                    time_limit=time_limit,
                )
                tdiff = time.process_time() - t
                results_sample[i] = {
                    "CE": sample_res,
                    "ll_threshold": min(median_ll, sample_ll),
                    "is_opt": opt,
                    "time": tdiff,
                }
                print(f"done sample {opt}, {len(sample_res)}")

                t = time.process_time()
                opt, nospn_res = lice.generate_counterfactual(
                    sample, not prediction, n_counterfactuals=10, time_limit=time_limit
                )
                tdiff = time.process_time() - t
                results_nospn[i] = {
                    "CE": nospn_res,
                    "ll_threshold": None,
                    "is_opt": opt,
                    "time": tdiff,
                }
                print(f"done nospn {opt}, {len(nospn_res)}")
            print(f"done iteration {i}")

        os.makedirs(f"{path_base}/CEs", exist_ok=True)

        if optimize:
            with open(f"{path_base}/CEs/LiCE_optimize.pickle", "wb") as f:
                pickle.dump(results_optimize, f)
        elif quartile:
            with open(f"{path_base}/CEs/LiCE_quartile.pickle", "wb") as f:
                pickle.dump(results_quartile, f)
        elif median:
            with open(f"{path_base}/CEs/LiCE_median.pickle", "wb") as f:
                pickle.dump(results_median, f)
        else:
            with open(f"{path_base}/CEs/LiCE_sample.pickle", "wb") as f:
                pickle.dump(results_sample, f)

            with open(f"{path_base}/CEs/MIO_no_spn.pickle", "wb") as f:
                pickle.dump(results_nospn, f)
