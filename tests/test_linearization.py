import pytest
import torch
from torch_geometric.utils import degree, sort_edge_index

from autograph.data.mol_dataset import QM9Dataset
from autograph.data.sent_utils_wrapper import (
    get_graph_from_labeled_sent,
    get_graph_from_sent,
    sample_labeled_sent_from_graph,
)
from autograph.linearization import STRATEGIES

# ==========================================
# 1. CONFIGURATION
# ==========================================

TOKENIZER_CONFIG = {
    "idx_offset": 10,
    "reset": 1,
    "ladj": 2,
    "radj": 3,
    "node_idx_offset": 100,
    "edge_idx_offset": 200,
    "num_node_types": 20,
    "num_edge_types": 10,
}

DATA_ROOT = "./datasets"

# ==========================================
# 2. VERIFICATION HELPERS
# ==========================================


def check_topological_invariant(data, node_index_map, strategy_name):
    """Verifies Hub/Leaf constraints (excluding random/hybrid)."""
    if strategy_name in ["random_order", "anchor_expansion"]:
        return

    deg = degree(data.edge_index[0], data.num_nodes)
    start_node_original = (node_index_map == 0).nonzero(as_tuple=True)[0].item()
    start_node_deg = deg[start_node_original].item()

    if strategy_name == "max_degree_first":
        expected = deg.max().item()
        assert start_node_deg == expected, (
            f"Max Degree Fail: Got {start_node_deg}, expected {expected}"
        )

    elif strategy_name == "min_degree_first":
        expected = deg.min().item()
        assert start_node_deg == expected, (
            f"Min Degree Fail: Got {start_node_deg}, expected {expected}"
        )


def check_reconstruction(data, sent_seq, node_index_map, is_labeled):
    """Full isomorphism check for connectivity and attributes."""
    recon_args = {**TOKENIZER_CONFIG, "undirected": True}

    if is_labeled:
        recon_edge_index, recon_x, recon_e = get_graph_from_labeled_sent(sent_seq, **recon_args)
    else:
        recon_edge_index = get_graph_from_sent(
            sent_seq,
            **{k: TOKENIZER_CONFIG[k] for k in ["idx_offset", "reset", "ladj", "radj"]},
            undirected=True,
        )

    map_tensor = node_index_map.long()
    orig_row, orig_col = data.edge_index
    mapped_edge_index = torch.stack([map_tensor[orig_row], map_tensor[orig_col]], dim=0)

    # Connectivity
    m_sorted, _ = sort_edge_index(mapped_edge_index)
    r_sorted, _ = sort_edge_index(recon_edge_index)
    assert torch.equal(m_sorted, r_sorted), "Edge index mismatch"

    if is_labeled:
        # Node Labels
        mapped_x = torch.zeros_like(data.x.flatten())
        mapped_x[map_tensor] = data.x.flatten()
        assert torch.equal(mapped_x, recon_x), "Node label mismatch"

        # Edge Attributes
        def get_canonical_edges(edge_index, edge_attr):
            edges = []
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                attr = edge_attr[i].item()
                edges.append((tuple(sorted((u, v))), attr))
            return sorted(edges)

        m_edges = get_canonical_edges(mapped_edge_index, data.edge_attr.flatten())
        r_edges = get_canonical_edges(recon_edge_index, recon_e.flatten())
        assert m_edges == r_edges, "Edge attribute mismatch"


# ==========================================
# 3. TEST CASES (Full Sweep)
# ==========================================

TEST_STRATEGIES = list(STRATEGIES.keys())


@pytest.mark.parametrize("strategy", TEST_STRATEGIES)
def test_full_qm9_linearization(strategy):
    """Exhaustive test on all molecular graphs in QM9 test split."""
    # Force download/load full test split
    dataset = QM9Dataset(root=DATA_ROOT, split="test")

    for _i, data in enumerate(dataset):
        x_in = data.x.flatten()
        e_in = (
            data.edge_attr.flatten()
            if data.edge_attr is not None
            else torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        )

        params = STRATEGIES[strategy].kwargs
        walk, node_map = sample_labeled_sent_from_graph(
            data.edge_index,
            x_in,
            e_in,
            deterministic=True,
            **{
                k: TOKENIZER_CONFIG[k]
                for k in [
                    "idx_offset",
                    "node_idx_offset",
                    "edge_idx_offset",
                    "reset",
                    "ladj",
                    "radj",
                ]
            },
            **params,
        )

        node_map_t = torch.from_numpy(node_map)
        check_topological_invariant(data, node_map_t, strategy)
        check_reconstruction(data, torch.from_numpy(walk), node_map_t, is_labeled=True)
