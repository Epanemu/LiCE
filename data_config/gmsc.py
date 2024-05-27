import pandas as pd

from LiCE.data.Features import Monotonicity

data = pd.read_csv("data/GMSC/cs-training.csv", index_col=0).dropna()
X = data[data.columns[1:]]
y = data[["SeriousDlqin2yrs"]]

# outlier pruning

mask = X["MonthlyIncome"] < 50000
mask &= X["RevolvingUtilizationOfUnsecuredLines"] < 1
mask &= X["NumberOfTime30-59DaysPastDueNotWorse"] < 10
mask &= X["DebtRatio"] < 2
mask &= X["NumberOfOpenCreditLinesAndLoans"] < 40
mask &= X["NumberOfTimes90DaysLate"] < 10
mask &= X["NumberRealEstateLoansOrLines"] < 10
mask &= X["NumberOfTime60-89DaysPastDueNotWorse"] < 10
mask &= X["NumberOfDependents"] < 10
X = X[mask]
y = y[mask]

# print(mask.mean())

# end outlier pruning

bounds = {"RevolvingUtilizationOfUnsecuredLines": (0, 1), "DebtRatio": (0, 2)}
for col in X.columns:
    if col not in bounds:
        bounds[col] = (X[col].min(), X[col].max())

config = {
    "X": X,
    "y": y,
    "categ_map": {},
    "ordered": [],
    "discrete": [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "age",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ],
    "bounds": bounds,
    "immutable": ["NumberOfDependents"],
    "monotonicity": {"age": Monotonicity.INCREASING},
    "causal_inc": [],
    "greater_than": [],
}
