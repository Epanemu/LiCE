# trunk-ignore-all(bandit/B403)
import copy
import os
import pickle
import sys
import time

import dice_ml
import pandas as pd
import torch.nn as tnn

import compared_methods.dice_ml_04.dice_ml as dice_ml_04  # downgraded version, heavily edited, local copy
from LiCE.data.Features import Categorical, Contiguous, Monotonicity
from nn_model import NNModel

# trunk-ignore-all(bandit/B301)
# trunk-ignore-all(ruff/B023)

data_names = [sys.argv[1]]
folds = [int(sys.argv[2])]
folder = sys.argv[3]

prefix = f"results/{folder}"

for fold in folds:
    for data_name in data_names:
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

        # PREPARATIONS

        nn_sigmoid = copy.deepcopy(nn)
        nn_sigmoid.model.append(tnn.Sigmoid())

        y_train.columns = [dhandler.target_feature.name]
        y_test.columns = [dhandler.target_feature.name]

        cont = [f.name for f in dhandler.features if isinstance(f, Contiguous)]
        train_data = pd.concat([X_train, y_train], axis=1)
        mutable = [f.name for f in dhandler.features if f.modifiable]

        # DICE
        d = dice_ml.Data(
            dataframe=train_data,
            continuous_features=cont,
            outcome_name=dhandler.target_feature.name,
        )
        backend = "PYT"

        one_hot_cols = []
        for f in dhandler.features:
            if isinstance(f, Categorical):
                for val in f.orig_vals:
                    one_hot_cols.append(f"{f.name}_{val}")
            else:
                one_hot_cols.append(f.name)

        def encode(sample, **kwargs):
            # return pd.DataFrame(
            #     dhandler.encode(sample.astype(int)), columns=one_hot_cols
            # )
            return pd.DataFrame(dhandler.encode(sample.values), columns=one_hot_cols)

        def decode(sample, **kwargs):
            return dhandler.decode(sample.values)

        # the inverse transformation is a workaround, that works only because I edited the dice library.
        # the change required is on line 329 (can change) of the dice_ml/utils/helpers.py file
        # in the function initialize_transform_func(self), if using custom function, add the inverse function as well
        # self.data_transformer = FunctionTransformer(func=self.func, inverse_func=self.kw_args["inverse_func"], kw_args=self.kw_args, validate=False)
        # above is the edited line, arrows point to the edit          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # link to github repo reference to the line https://github.com/interpretml/DiCE/blob/48832802c2a0822a9b203f3057e6def9e8ba0d0a/dice_ml/utils/helpers.py#L329C12-L329C12
        kw_args = {}
        kw_args["inverse_func"] = decode
        backend = "PYT"
        m = dice_ml.Model(
            model=nn_sigmoid.model, backend=backend, func=encode, kw_args=kw_args
        )
        exp = dice_ml.Dice(d, m, method="gradient")

        # VAE
        d04 = dice_ml_04.Data(
            dataframe=train_data,
            continuous_features=cont,
            outcome_name=dhandler.target_feature.name,
            data_handler=dhandler,
            one_hot_cols=one_hot_cols,
            data_name=f"{data_name}-{fold}",
        )

        backend = {
            "model": "pytorch_model.PyTorchModel",
            "explainer": "feasible_base_vae.FeasibleBaseVAE",
        }
        m04 = dice_ml_04.Model(
            model=nn_sigmoid.model, backend=backend, data_handler=dhandler
        )
        exp04 = dice_ml_04.Dice(
            d04,
            m04,
            encoded_size=len(one_hot_cols),
            lr=1e-3,
            batch_size=64,
            validity_reg=10,
            margin=0.1,
            epochs=20 if X_train.shape[0] > 1000 else 200,
            wm1=1e-4,
            wm2=1e-4,
            wm3=1e-4,
        )
        exp04.train()

        results_dice = {}
        results_vae = {}
        for i, sample in X_test.iterrows():
            enc_sample = dhandler.encode(X_test.loc[[i]])
            prediction = nn.predict(enc_sample) > 0

            ranges = {}
            for f in dhandler.features:
                if f.name in cont:
                    lb, ub = f.bounds
                    if f.monotone == Monotonicity.INCREASING:
                        lb = f.encode(sample[f.name])
                    elif f.monotone == Monotonicity.DECREASING:
                        ub = f.encode(sample[f.name])

                    ranges[f.name] = (lb, ub)

            t = time.process_time()
            try:
                dice_res = exp.generate_counterfactuals(
                    pd.DataFrame([sample]),
                    total_CFs=10,
                    desired_class=int(not prediction),
                    features_to_vary=mutable,
                    permitted_range=ranges,
                )
            except Exception as e:
                print(e)
                dice_res = None
            tdiff = time.process_time() - t
            results_dice[i] = {
                "CE": (
                    []
                    if dice_res is None
                    else dice_res.cf_examples_list[0].final_cfs_df.values
                ),
                "ll_threshold": None,
                "is_opt": False,
                "time": tdiff,
            }

            t = time.process_time()
            vae_res = exp04.generate_counterfactuals(
                pd.DataFrame([sample]),
                total_CFs=10,
                desired_class=int(not prediction),
            )
            tdiff = time.process_time() - t
            results_vae[i] = {
                "CE": [] if vae_res is None else vae_res.final_cfs_df.values,
                "ll_threshold": None,
                "is_opt": False,
                "time": tdiff,
            }

            print("One round done")
        os.makedirs(f"{path_base}/CEs", exist_ok=True)

        with open(f"{path_base}/CEs/DiCE.pickle", "wb") as f:
            pickle.dump(results_dice, f)

        with open(f"{path_base}/CEs/VAE.pickle", "wb") as f:
            pickle.dump(results_vae, f)
