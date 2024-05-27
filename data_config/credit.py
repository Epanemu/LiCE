import numpy as np
from ucimlrepo import fetch_ucirepo

from LiCE.data.Features import Monotonicity

credit_data = fetch_ucirepo(id=144)

X = credit_data.data.features
y = credit_data.data.targets
col_mapping = {
    row["name"]: row["description"] for _, row in credit_data.variables.iterrows()
}
X.rename(columns=col_mapping, inplace=True)
y.replace({1: "Good", 2: "Bad"}, inplace=True)

mask = ~np.any(X.isna(), axis=1)
X = X[mask]
y = y[mask]


print(credit_data.data.features.shape)
print(X.shape)
print(credit_data.data.features.shape[0] - X.shape[0])
print(sum(mask))

config = {
    "X": X,
    "y": y,
    "categ_map": {
        "Status of existing checking account": [],  # order?
        "Credit history": [],
        "Purpose": [],
        "Savings account/bonds": [],  # order?
        "Present employment since": ["A71", "A72", "A73", "A74", "A75"],
        "Personal status and sex": [],
        "Other debtors / guarantors": [],
        "Property": [],
        "Other installment plans": [],
        "Housing": [],
        "Job": [],
        "Telephone": [],
        "foreign worker": [],
    },
    "ordered": ["Present employment since"],
    "discrete": [
        "Duration",
        "Present residence since",
        "Age",
        "Number of existing credits at this bank",
        "Number of people being liable to provide maintenance for",
    ],
    "bounds": {
        "Age": (0, 100),
        "Installment rate in percentage of disposable income": (0, 100),
        "Duration": (X["Duration"].min(), X["Duration"].max()),
        "Credit amount": (X["Credit amount"].min(), X["Credit amount"].max()),
        "Present residence since": (
            X["Present residence since"].min(),
            X["Present residence since"].max(),
        ),
        "Number of existing credits at this bank": (
            X["Number of existing credits at this bank"].min(),
            X["Number of existing credits at this bank"].max(),
        ),
        "Number of people being liable to provide maintenance for": (
            X["Number of people being liable to provide maintenance for"].min(),
            X["Number of people being liable to provide maintenance for"].max(),
        ),
    },
    "immutable": [
        "Number of people being liable to provide maintenance for",
        "Personal status and sex",
        "foreign worker",
    ],
    "monotonicity": {"Age": Monotonicity.INCREASING},
    "causal_inc": [
        ("Present residence since", "Age"),
        ("Present employment since", "Age"),
    ],
    "greater_than": [],
}
