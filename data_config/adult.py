import numpy as np
from ucimlrepo import fetch_ucirepo

from LiCE.data.Features import Monotonicity

adult_data = fetch_ucirepo(id=2)

X = adult_data.data.features
y = adult_data.data.targets

X = X.drop(
    columns=[
        "fnlwgt",
        "education-num",
        "native-country",
        "capital-gain",
        "capital-loss",
    ]
)
mask = ~np.any(X.isna(), axis=1)
X = X[mask]
y = y[mask].replace({"<=50K.": "<=50K", ">50K.": ">50K"})

edu_level = (
    adult_data.data.features[["education-num", "education"]]
    .groupby("education-num")
    .agg(max)
)
edu_levels = [edu_level.loc[i]["education"] for i in sorted(edu_level.index)]

bounds = {"RevolvingUtilizationOfUnsecuredLines": (0, 1), "DebtRatio": (0, 1)}
for col in X.columns:
    if col not in bounds:
        bounds[col] = (X[col].min(), X[col].max())

config = {
    "X": X,
    "y": y,
    "categ_map": {
        "education": edu_levels,
        "marital-status": [],
        "occupation": [],
        "relationship": [],
        "workclass": [],
        "race": [],
        "sex": [],
    },
    "ordered": ["education"],
    "discrete": ["hours-per-week", "age"],
    "bounds": {"hours-per-week": (0, 99)},
    "immutable": ["race", "sex"],
    "monotonicity": {
        "age": Monotonicity.INCREASING,
        "education": Monotonicity.INCREASING,
    },
    "causal_inc": [("education", "age")],
    "greater_than": [],
}
