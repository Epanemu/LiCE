import os
import pickle

import bnlearn
import numpy as np
import pandas as pd
from spn.algorithms.Inference import likelihood, log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


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


def compare_on_BN(name, meta_types, n_train, n_test, runs):
    model = bnlearn.import_DAG(name)
    name = name.split("/")[-1].split(".")[0]
    CPD = bnlearn.print_CPD(model, verbose=0)

    domains = {}
    for k, df in CPD.items():
        domains[k] = df[k].unique()
    print(domains)

    for seed in range(runs):
        print(f"{name} RUN {seed+1}/{runs}")
        # sample
        np.random.seed(seed)
        train_data = bnlearn.sampling(model, n_train, verbose=False)
        test_data = bnlearn.sampling(model, n_test, verbose=False)

        # fit SPN and comp log likelihood
        d_context = Context(
            meta_types=meta_types, domains=[domains[c] for c in train_data.columns]
        )
        spn = learn_mspn(train_data.to_numpy(), d_context)
        ll = log_likelihood(spn, test_data.to_numpy())

        # Comp BN probability
        evaluated = {}
        ps = []
        for i, d in test_data.iterrows():
            h = " ".join([f"{d[k]}" for k in test_data.columns])
            if h in evaluated:
                p = evaluated[h]
            else:
                p = compute_p(CPD, d)
                evaluated[h] = p
            ps.append(p)
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{n_test} done", flush=True)

        lps = np.log(ps)
        os.makedirs(f"results/bayes_nets/SPN_fit/{name}/", exist_ok=True)
        with open(f"results/bayes_nets/SPN_fit/{name}/ll_lps_{seed}.pickle", "wb") as f:
            pickle.dump((ll, lps), f)

        # visualize and evaluate the results in 3_visualize_BN_results.ipynb
        if MetaType.REAL not in meta_types and len(meta_types) < 12:

            def recurse_tv(col_pos, assignment, cols, domains, spn, CPD):
                if col_pos >= len(domains):
                    diff = abs(
                        likelihood(spn, assignment.to_numpy().reshape((1, -1)))[0, 0]
                        - compute_p(CPD, assignment)
                    )
                    return diff

                total = 0
                k = cols[col_pos]
                for val in domains[k]:
                    assignment[k] = val
                    total += recurse_tv(
                        col_pos + 1, assignment, cols, domains, spn, CPD
                    )
                return total

            empty_vec = pd.Series(
                [np.nan] * len(domains), index=train_data.columns, dtype=float
            )
            tv = recurse_tv(0, empty_vec, train_data.columns, domains, spn, CPD) / 2
            print(f"{name} ({seed}) TOTAL VARIATION: {tv}")
            with open(f"results/bayes_nets/SPN_fit/{name}/TV_{seed}.txt", "w") as f:
                f.write(f"{tv}")


if __name__ == "__main__":
    # this script should be run from the top level folder
    train_samples = 10000
    test_samples = 1000
    runs = 5
    compare_on_BN("sprinkler", [MetaType.BINARY] * 4, train_samples, test_samples, runs)
    compare_on_BN("asia", [MetaType.BINARY] * 8, train_samples, test_samples, runs)
    compare_on_BN("sachs", [MetaType.DISCRETE] * 11, train_samples, test_samples, runs)
    compare_on_BN(
        "data/BayesianNetworks/child.bif",
        [MetaType.DISCRETE] * 20,
        train_samples,
        test_samples,
        runs,
    )
    compare_on_BN("water", [MetaType.DISCRETE] * 32, train_samples, test_samples, runs)
    compare_on_BN("alarm", [MetaType.DISCRETE] * 37, train_samples, test_samples, runs)
    compare_on_BN(
        "data/BayesianNetworks/win95pts.bif",
        [MetaType.BINARY] * 76,
        train_samples,
        test_samples,
        runs,
    )
    compare_on_BN("andes", [MetaType.BINARY] * 223, train_samples, test_samples, runs)
