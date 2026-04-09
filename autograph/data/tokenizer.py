from typing import ClassVar

import torch
from torch_geometric.data import Data

from autograph.linearization import STRATEGIES

from .batch_converter import BatchConverter
from .sent_utils_wrapper import (
    get_graph_from_labeled_sent,
    get_graph_from_sent,
    sample_labeled_sent_from_graph,
    sample_sent_from_graph,
)


class Graph2TrailTokenizer:
    sos: int = 0
    reset: int = 1
    ladj: int = 2
    radj: int = 3
    eos: int = 4
    pad: int = 5
    special_toks: ClassVar[list[str]] = ["sos", "reset", "ladj", "radj", "eos", "pad"]

    def __init__(
        self,
        dataset_names=None,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        append_eos=True,
        rng=None,
        linearization_strategy="random_order",
        deterministic=False,
        **kwargs,
    ):
        self.dataset_names = dataset_names if dataset_names is not None else []
        self.max_length = max_length
        self.undirected = undirected
        self.append_eos = append_eos
        self.truncation_length = truncation_length
        self.rng = rng
        self.deterministic = deterministic

        if linearization_strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown linearization strategy: {linearization_strategy}. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        self.linearization_strategy = linearization_strategy
        self.strategy_params = STRATEGIES[linearization_strategy].kwargs

        # Initialize offsets with safe defaults to prevent AttributeErrors
        self.max_num_nodes = 1000
        self.num_node_types = 0
        self.num_edge_types = 0
        self.idx_offset = len(self.special_toks) + len(self.dataset_names)
        self.node_idx_offset = self.idx_offset + self.max_num_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

        if len(self.dataset_names) > 0:
            self.dataset_to_idx = {
                dataset_name: i + len(self.special_toks)
                for i, dataset_name in enumerate(self.dataset_names)
            }
        self.idx_offset = len(self.special_toks) + len(self.dataset_names)
        self.max_num_nodes = None
        self.labeled_graph = labeled_graph
        self.num_node_types = self.num_edge_types = 0

    def set_num_nodes(self, max_num_nodes):
        if (self.max_num_nodes is None) or (self.max_num_nodes < max_num_nodes):
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        max_nodes = self.max_num_nodes if self.max_num_nodes is not None else 1000
        self.node_idx_offset = self.idx_offset + max_nodes
        self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    def __len__(self):
        max_nodes = self.max_num_nodes if self.max_num_nodes is not None else 1000
        if self.labeled_graph:
            return self.idx_offset + max_nodes + self.num_node_types + self.num_edge_types
        return self.idx_offset + max_nodes

    def __call__(self, data):
        return self.tokenize(data)

    def get_dataset_idx(self, dataset_name):
        return self.dataset_to_idx.get(dataset_name, None)

    def tokenize(self, data):
        if not data.is_coalesced():
            data = data.coalesce()

        common_kwargs = {
            "num_nodes": data.num_nodes,
            "max_length": self.max_length,
            "idx_offset": self.idx_offset,
            "reset": self.reset,
            "ladj": self.ladj,
            "radj": self.radj,
            "undirected": self.undirected,
            "rng": self.rng,
            "deterministic": self.deterministic,
            **self.strategy_params,
        }

        if self.labeled_graph:
            walk_index, _ = sample_labeled_sent_from_graph(
                edge_index=data.edge_index,
                node_labels=data.x.flatten(),
                edge_labels=data.edge_attr,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                **common_kwargs,
            )
        else:
            walk_index, _ = sample_sent_from_graph(edge_index=data.edge_index, **common_kwargs)

        start_offset = 1  # sos
        end_offset = 0
        if self.append_eos:
            end_offset += 1  # eos
        dataset_name_idx = None
        if hasattr(data, "dataset_name") and len(self.dataset_names) > 0:
            dataset_name_idx = self.get_dataset_idx(data.dataset_name)
            if dataset_name_idx is not None:
                start_offset += 1

        walk_index = torch.from_numpy(walk_index)

        walk_index_new = torch.zeros(
            (walk_index.shape[0] + start_offset + end_offset,), dtype=walk_index.dtype
        )
        walk_index_new[0] = self.sos
        if dataset_name_idx is not None:
            walk_index_new[1] = dataset_name_idx
        if self.append_eos:
            walk_index_new[-1] = self.eos
            walk_index_new[start_offset:-1] = walk_index
        else:
            walk_index_new[start_offset:] = walk_index

        return walk_index_new

    def decode(self, walk_index):
        walk_index = walk_index[
            (walk_index != self.pad) & (walk_index != self.sos) & (walk_index != self.eos)
        ]
        dataset_name = None
        if len(walk_index) > 0 and len(self.dataset_names) > 0:
            ds_idx = walk_index[0].item() - len(self.special_toks)
            if 0 <= ds_idx < len(self.dataset_names):
                dataset_name = self.dataset_names[ds_idx]
                walk_index = walk_index[1:]

        if len(walk_index) == 0:
            if self.labeled_graph:
                return Data(
                    x=torch.zeros((0,), dtype=torch.long),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0,), dtype=torch.long),
                    num_nodes=0,
                    dataset_name=dataset_name,
                )
            return Data(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                dataset_name=dataset_name,
                num_nodes=0,
            )

        if self.labeled_graph:
            edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
                walk_index=walk_index,
                idx_offset=self.idx_offset,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
                max_nodes=self.max_num_nodes,
            )
            num_nodes = 0
            if edge_index.numel() > 0:
                num_nodes = max(edge_index.flatten().max().item() + 1, len(node_labels))
            elif len(node_labels) > 0:
                num_nodes = len(node_labels)

            return Data(
                x=node_labels,
                edge_index=edge_index,
                edge_attr=edge_labels,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
            )
        edge_index = get_graph_from_sent(
            walk_index=walk_index,
            idx_offset=self.idx_offset,
            reset=self.reset,
            ladj=self.ladj,
            radj=self.radj,
            undirected=self.undirected,
        )
        return Data(
            edge_index=edge_index,
            dataset_name=dataset_name,
            num_nodes=edge_index.flatten().max().item() + 1 if edge_index.numel() > 0 else 0,
        )

    def batch_converter(self):
        return BatchConverter(self, self.truncation_length)
