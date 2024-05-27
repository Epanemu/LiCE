import numpy as np
import pyomo.environ as pyo

from ..spn.SPN import SPN, NodeType

# issues with binding variables in lambda functions for constraints
# trunk-ignore-all(ruff/B023)


def encode_spn(
    spn: SPN,
    mio_spn: pyo.Block,
    input_vars: list[list[pyo.Var] | pyo.Var],
    mio_epsilon: float = 0.0001,
) -> pyo.Var:
    """
    Encodes the spn into MIP formulation, computing log-likelihood over the input variables
    returns all variables

    input_vars either take the value to be inputed into spn, or they are a list of variables as one-hot encoded input
    in the same ordering as SPN bins

    mio_epsilon is the minimal change between values - for numerical stability - used here for sharp inequalities
    """
    node_ids = [node.id for node in spn.nodes]
    # node_type_ids = {t: [] for t in NodeType}
    # for node in spn.nodes:
    #     node_type_ids[node.type].append(node.id)
    #     node_ids.append(node.id)

    # mio_spn.node_type_sets = {
    #     t: pyo.Set(initialize=ids) for t, ids in node_type_ids.items()
    # }
    mio_spn.node_set = pyo.Set(initialize=node_ids)

    # values are log likelihoods - always negative - except in narrow peaks that go above 1
    # mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.NonPositiveReals)
    mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.Reals)
    # print(mio_spn.node_set, node_ids)

    # TODO nodes as blocks
    for node in spn.nodes:
        if node.type == NodeType.LEAF:
            # in_var = mio_spn.input[node.scope]
            in_var = input_vars[node.scope]

            breakpoints = [node.breaks[0]]
            for b in node.breaks[1:-1]:
                breakpoints += [b, b]
            breakpoints.append(node.breaks[-1])

            density_vals = []
            for d in node.densities:
                density_vals += [d, d]

            lb, ub = in_var.bounds

            if lb is None or ub is None:
                raise AssertionError("SPN input variables must have fixed bounds.")
            # if histogram is narrower than the input bounds
            # if lb + mio_epsilon < breakpoints[0]:
            if lb < breakpoints[0]:
                breakpoints = [lb, breakpoints[0]] + breakpoints
                density_vals = [spn.min_density, spn.min_density] + density_vals
            # if ub - mio_epsilon > breakpoints[-1]:
            if ub > breakpoints[-1]:
                breakpoints = breakpoints + [breakpoints[-1], ub]
                density_vals = density_vals + [spn.min_density, spn.min_density]

            # possibly further removal of too similar bins
            # new_vals = [density_vals[0]]
            # new_breaks = [breakpoints[0]]
            # for i in range(2, len(density_vals), 2):
            #     if np.abs(new_vals[-1] - density_vals[i]) >= mio_epsilon:
            #         new_vals.append(density_vals[i - 1])
            #         new_vals.append(density_vals[i])
            #         new_breaks.append(breakpoints[i - 1])
            #         new_breaks.append(breakpoints[i])
            # density_vals = new_vals
            # breakpoints = new_breaks
            # i = 0
            # while i < len(density_vals) - 2:
            #     if np.abs(density_vals[i] - density_vals[i + 2]) <= mio_epsilon:
            #         density_vals.pop(i + 1)
            #         density_vals.pop(i + 1)
            #         breakpoints.pop(i + 1)
            #         breakpoints.pop(i + 1)
            #     else:
            #         i += 2
            log_densities = np.log(density_vals)
            # print(list(log_densities), len(log_densities))
            # print(breakpoints, len(breakpoints))

            pw_constr = pyo.Piecewise(
                # mio_spn.node_out[i],
                mio_spn.node_out[node.id],
                in_var,
                pw_pts=breakpoints,
                pw_constr_type="EQ",
                f_rule=list(log_densities),
            )
            mio_spn.add_component(f"PWLeaf{node.id}", pw_constr)
            # TODO test the effectiveness of this
            # TODO filter the values

        elif node.type == NodeType.LEAF_CATEGORICAL:
            dens_ll = np.log(node.densities)
            in_vars = input_vars[node.scope]

            if isinstance(in_vars, pyo.Var):
                in_vars = [in_vars[k] for k in sorted(in_vars.keys())]

            if len(in_vars) <= 1:  # TODO make this more direct, not fixed to 1
                raise ValueError(
                    "The categorical values should be passed as a list of binary variables, representing a one-hot encoding."
                )
            # Do checks that the vars are binary?
            # check if the histogram always contains all values?
            # TODO use expr parameter of Constraint maker, instead of the rule=lambdas?

            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == sum(var * dens for var, dens in zip(in_vars, dens_ll))
                )
            )
            mio_spn.add_component(f"CategLeaf{node.id}", constr)
        elif node.type == NodeType.LEAF_BINARY:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == (1 - input_vars[node.scope]) * np.log(node.densities[0])
                    + input_vars[node.scope] * np.log(node.densities[1])
                )
            )
            mio_spn.add_component(f"BinLeaf{node.id}", constr)
        elif node.type == NodeType.PRODUCT:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == sum(b.node_out[ch.id] for ch in node.predecessors)
                )
            )
            mio_spn.add_component(f"ProdConstr{node.id}", constr)
        elif node.type == NodeType.SUM:
            # Sum node - approximated in log domain by max
            # implemented using SOS1 constraints, see here: https://www.gurobi.com/documentation/current/refman/general_constraints.html
            preds_set = [ch.id for ch in node.predecessors]
            weights = {ch.id: w for ch, w in zip(node.predecessors, node.weights)}

            slacks = pyo.Var(preds_set, domain=pyo.NonNegativeReals)
            mio_spn.add_component(f"SumSlackVars{node.id}", slacks)
            slacking = pyo.Constraint(
                preds_set,
                rule=lambda b, pre_id: (
                    b.node_out[node.id]
                    == b.node_out[pre_id] + np.log(weights[pre_id]) + slacks[pre_id]
                ),
            )
            mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)

            indicators = pyo.Var(preds_set, domain=pyo.Binary)
            mio_spn.add_component(f"SumIndicators{node.id}", indicators)
            indicating = pyo.Constraint(
                rule=lambda b: (
                    sum(b.component(f"SumIndicators{node.id}")[i] for i in preds_set)
                    == 1
                )
            )
            mio_spn.add_component(f"SumIndicatorConstr{node.id}", indicating)

            sos = pyo.SOSConstraint(
                preds_set,
                rule=lambda b, pred: [
                    b.component(f"SumIndicators{node.id}")[pred],
                    b.component(f"SumSlackVars{node.id}")[pred],
                ],
                sos=1,
            )
            mio_spn.add_component(f"SumSosConstr{node.id}", sos)

    return mio_spn.node_out
