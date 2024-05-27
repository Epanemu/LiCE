import numpy as np
import onnx
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io import load_onnx_neural_network
from omlt.neuralnet import FullSpaceNNFormulation
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverStatus, TerminationCondition

from LiCE.data.DataHandler import DataHandler
from LiCE.data.Features import Contiguous
from LiCE.data.Types import DataLike
from LiCE.spn.SPN import SPN

from .data_enc import decode_input_change, encode_input_change
from .spn_enc import encode_spn


class LiCE:
    def __init__(self, spn: SPN, nn_path: str, data_handler: DataHandler) -> None:
        self.__spn = spn
        self.__nn_path = nn_path
        self.__dhandler = data_handler

    def __build_model(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float,
        optimize_ll: bool,
        prediction_threshold: float = 1e-4,
        ll_opt_coef: float = 0.1,
    ) -> pyo.Model:
        model = pyo.ConcreteModel()

        model.input_encoding = pyo.Block()
        inputs, distance = encode_input_change(
            self.__dhandler, model.input_encoding, factual
        )

        model.predictor = OmltBlock()
        onnx_model = onnx.load(self.__nn_path)
        input_bounds = []
        input_vec = []
        for input_var in inputs:
            for var in input_var.values():
                input_vec.append(var)
                input_bounds.append(var.bounds)

        net = load_onnx_neural_network(onnx_model, input_bounds=input_bounds)
        formulation = FullSpaceNNFormulation(net)
        model.predictor.build_formulation(formulation)

        # connect the vars
        model.inputset = pyo.Set(initialize=range(len(input_vec)))

        def connect_input(mdl, i):
            return input_vec[i] == mdl.predictor.inputs[i]

        model.connect_nn_input = pyo.Constraint(model.inputset, rule=connect_input)

        sign = -1 if desired_class == 0 else 1
        model.classification = pyo.Constraint(
            expr=sign * model.predictor.outputs[0] >= prediction_threshold
        )

        # TODO put this to dataenc or to spn, using the fact that spn object knows about features (afaik)
        spn_inputs = []
        model.contig_names = pyo.Set(
            initialize=[
                f.name for f in self.__dhandler.features if isinstance(f, Contiguous)
            ]
        )
        contig_bounds = {
            f.name: f.bounds
            for f in self.__dhandler.features
            if isinstance(f, Contiguous)
        }
        model.spn_input = pyo.Var(
            model.contig_names, bounds=contig_bounds, domain=pyo.Reals
        )

        def set_scale(m, name: str):
            i = self.__dhandler.feature_names.index(name)
            f = self.__dhandler.features[i]
            return m.spn_input[name] == inputs[i] * f._scale + f._shift

        model.spn_input_set = pyo.Constraint(model.contig_names, rule=set_scale)
        for input_var, f in zip(inputs, self.__dhandler.features):
            if isinstance(f, Contiguous):
                spn_inputs.append(model.spn_input[f.name])
            else:
                spn_inputs.append(input_var)

        if optimize_ll:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(self.__spn, model.spn, inputs)
            model.obj = pyo.Objective(
                expr=distance - ll_opt_coef * spn_outputs[self.__spn.out_node_id],
                sense=pyo.minimize,
            )
            return model

        elif ll_threshold > -np.inf:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn, model.spn, spn_inputs + [int(desired_class)]
            )
            model.ll_constr = pyo.Constraint(
                expr=spn_outputs[self.__spn.out_node_id] >= ll_threshold
            )

        # set up objective
        model.obj = pyo.Objective(expr=distance, sense=pyo.minimize)
        # model.objconstr = pyo.Constraint(expr=distance == 0)
        # model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        return model

    def generate_counterfactual(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float = -np.inf,
        optimize_ll: bool = False,
        n_counterfactuals: int = 1,
        solver_name: str = "gurobi",
        verbose: bool = False,
        time_limit: int = 600,
    ) -> tuple[bool, list[DataLike]]:

        model = self.__build_model(factual, desired_class, ll_threshold, optimize_ll)
        opt = pyo.SolverFactory(solver_name, solver_io="python")

        if n_counterfactuals > 1:
            if solver_name != "gurobi":
                raise NotImplementedError(
                    "Generating multiple counterfactuals is supported only for Gurobi solver"
                )
            opt.options["PoolSolutions"] = n_counterfactuals  # Store 10 solutions
            opt.options["PoolSearchMode"] = 2  # Systematic search for k-best solutions
            # opt.options['PoolGap'] = 0.1       # Accept solutions within 10% of the optimal

        if solver_name == "gurobi":
            opt.options["TimeLimit"] = time_limit
        else:
            print("Time limit not set!")

        result = opt.solve(model, load_solutions=True, tee=verbose)

        if verbose:
            print(result)
        if result.solver.status == SolverStatus.ok:
            if result.solver.termination_condition == TerminationCondition.optimal:
                # print(pyo.value(model.obj))
                # print(model.spn.node_out[self.__spn.out_node_id].value)
                CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
                return True, CEs
        elif result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            print("Infeasible formulation")
            # if verbose:
            write_iis(model, "IIS.ilp", solver="gurobi")
            return False, []
        elif (
            result.solver.status == SolverStatus.aborted
            and result.solver.termination_condition == TerminationCondition.maxTimeLimit
        ):
            print("TIME LIMIT")
            CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
            return False, CEs
        # else:

        # print result if it wasn't printed yet
        if not verbose:
            print(result)
        # write_iis(model, "IIS.ilp", solver="gurobi")
        raise ValueError("Unexpected termination condition")

    def __get_CEs(
        self, n: int, model: pyo.Model, factual: np.ndarray, opt: pyo.SolverFactory
    ):
        if n > 1:
            CEs = []
            for sol in range(min(n, opt._solver_model.SolCount)):
                opt._solver_model.params.SolutionNumber = sol
                vars = opt._solver_model.getVars()
                for var in vars:
                    opt._solver_var_to_pyomo_var_map[var].value = np.round(
                        var.Xn, 12  # correct some numerical errors
                    )
                CEs.append(
                    decode_input_change(self.__dhandler, model.input_encoding, factual)
                )
            return CEs
        else:
            return [decode_input_change(self.__dhandler, model.input_encoding, factual)]
