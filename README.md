# Likely Counterfactual Explanations (LiCE)
This is a Python implementation of the LiCE method introduced in the paper _Generating Likely Counterfactuals Using Sum-Product Networks_.

The method implementation itself is in the `LiCE` directory, including the implementation of the Mixed polytope encoding.

Example usage of LiCE method can be performed using the set of command at the end of the README.

## Data
The folders `data` contains data of the GMSC dataset (rest is accessed using a library), and `data_config` contains the configuration of each specific dataset.

To run LiCE on your dataset, first set up a configuration for it, similar to the ones in `data_config`.

## Usage
Requirements to run LiCE are in `requirements.txt` file.
LiCE uses the [Gurobi solver](http://www.gurobi.com/) for the MIP solver. This can be changed in the `solver_name` parameter to the method `generate_counterfactual`, however some gurobi-specific settings will not work.

```sh
conda create -n "LiCE_env" python==3.11
conda activate LiCE_env
pip install -r requirements.txt # assuming working directory is this one
# prepare data splits and models (NN, SPN...)
python data_prep.py
# rerun the experiments - requires setting up gurobi license

# python compute_CEs.py complete_results 120 {credit,adult,gmsc} {0..4} {median,quartile,optimize,none}
python compute_CEs.py complete_results 120 credit 0 median # e.g.

# results will be in ./results/complete_results folder
```
