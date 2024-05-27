from enum import Enum
from typing import Any

import numpy as np
from spn.algorithms.Inference import EPSILON, log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context, Leaf
from spn.structure.Base import Node as SPFlow_Node
from spn.structure.Base import Product, Sum, get_topological_order
from spn.structure.StatisticalTypes import MetaType

from LiCE.data.DataHandler import DataHandler
from LiCE.data.Features import Binary, Categorical, Contiguous, Feature, Mixed
from LiCE.data.Types import DataLike


class NodeType(
    Enum
):  # TODO make this into a class and subclasses so that isinstance leaf works on all 3 kinds of leaves
    SUM = 0
    PRODUCT = 1
    LEAF = 2
    LEAF_CATEGORICAL = 3
    LEAF_BINARY = 4


class Node:
    """A representation of a node in an SPN"""

    def __init__(
        self,
        node: SPFlow_Node,
        feature_list: list[Feature],
        normalize=True,  # pointless
        one_hot=False,  # pointless
    ):
        if isinstance(node, Leaf):
            self.densities = list(node.densities)
            if isinstance(node.scope, list):
                if len(node.scope) > 1:
                    raise NotImplementedError("Multivariate leaves are not supported.")
                self.scope = node.scope[0]
            else:
                self.scope = node.scope
            if isinstance(feature_list[self.scope], Categorical):
                self.type = NodeType.LEAF_CATEGORICAL
                self.options = feature_list[self.scope].numeric_vals
            elif isinstance(feature_list[self.scope], Binary):
                self.type = NodeType.LEAF_BINARY
            else:
                self.type = NodeType.LEAF
                # print(node.id, node.breaks, node.densities)
                self.discrete = feature_list[self.scope].discrete
                if self.discrete:
                    self.breaks = [b - 0.5 for b in node.breaks]
                else:
                    self.breaks = list(node.breaks)
                dens = node.densities
                duplicate = np.isclose(dens[1:], dens[:-1], atol=1e-10)
                self.densities = [dens[0]] + list(np.array(dens[1:])[~duplicate])
                self.breaks = (
                    [self.breaks[0]]
                    + list(np.array(self.breaks[1:-1])[~duplicate])
                    + [self.breaks[-1]]
                )
                # pruned_dens = [self.densities[0]]
                # pruned_breaks = [self.breaks[0]]
                # for i, d in enumerate(self.densities):
                #     if not np.isclose(pruned_dens[-1], d, atol=1e-10):
                #         pruned_dens.append(d)
                #         pruned_breaks.append(self.breaks[i])
                # self.densities = pruned_dens
                # self.breaks = pruned_breaks
        elif isinstance(node, Product):
            self.type = NodeType.PRODUCT
        elif isinstance(node, Sum):
            self.type = NodeType.SUM
            self.weights = node.weights
        else:
            raise ValueError("")
        self.name = node.name
        self.id = node.id
        # TODO make the predecessors also of this class, not the spflow one
        # TODO rework this so that the nodes are remembered in the SPN class, and not generated on demand
        self.predecessors = node.children if hasattr(node, "children") else []


class SPN:
    def __init__(
        self,
        data: DataLike,
        data_handler: DataHandler,
        normalize_data: bool = False,
        # trunk-ignore(ruff/B006)
        learn_mspn_kwargs: dict[str, Any] = {},
    ):
        types = []
        domains = []
        self.__feature_list = data_handler.features + [data_handler.target_feature]
        for feature in self.__feature_list:
            if isinstance(feature, Contiguous):
                if feature.discrete:
                    types.append(MetaType.DISCRETE)
                    domains.append(np.arange(feature.bounds[0], feature.bounds[1] + 1))
                else:
                    types.append(MetaType.REAL)
                    domains.append(np.asarray(feature.bounds))
            elif isinstance(feature, Categorical):
                types.append(MetaType.DISCRETE)
                domains.append(np.asarray(feature.numeric_vals))
            elif isinstance(feature, Binary):
                types.append(MetaType.BINARY)
                domains.append(np.asarray([0, 1]))
            elif isinstance(feature, Mixed):
                types.append(MetaType.REAL)
                domains.append(np.asarray(feature.bounds))
                # types.append(MetaType.DISCRETE) TODO add the doubling version to the mixed feature
            else:
                raise ValueError(f"Unsupported feature type of feature {feature}")

        # TODO: add domain handling to the vars
        # TODO: parametric types - types with distributions attached
        context = Context(
            meta_types=types,
            domains=domains,
            feature_names=[f.name for f in self.__feature_list],
        )
        self.__normalize_data = normalize_data
        enc_data = data_handler.encode_all(
            data, normalize=normalize_data, one_hot=False
        )
        if len(domains) != data_handler.n_features + 1:
            print("recomputing domains")
            context.add_domains(enc_data)

        self.__data_handler = data_handler
        self.__mspn = learn_mspn(enc_data, context, **learn_mspn_kwargs)
        self.__nodes = [
            Node(node, self.__feature_list)
            for node in get_topological_order(self.__mspn)
        ]

    def compute_ll(self, data: DataLike):
        if len(data.shape) == 1:
            return self.compute_ll(data.reshape(1, -1))[0]
        return log_likelihood(
            self.__mspn,
            self.__data_handler.encode_all(
                data, normalize=self.__normalize_data, one_hot=False
            ),
        )

    @property
    def nodes(self) -> list[Node]:
        if not hasattr(self, "SPN__nodes"):
            self.__nodes = [
                Node(node, self.__feature_list)
                for node in get_topological_order(self.__mspn)
            ]
        return self.__nodes

    @property
    def min_density(self) -> float:
        return EPSILON

    @property
    def out_node_id(self) -> float:
        return self.__mspn.id

    @property
    def spn_model(self):
        return self.__mspn
