"""
graph_builder.py

Converts a numeric connectivity matrix (or a sequence of them) into
PyTorch Geometric `Data` graphs.

Graph definition:
    nodes      -> ROIs. Node features = that ROI's full row in the
                  connectivity matrix (its connectivity profile to every
                  other ROI), giving each node an R-dimensional feature
                  vector.
    edges      -> the EDGE_TOP_K strongest connections (by |weight|) per
                  node, made bidirectional.
    edge_attr  -> the connectivity weight for that edge.

Tested against: Python 3.9.0
"""

from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

import config


def matrix_to_graph(matrix: np.ndarray, top_k: int = config.EDGE_TOP_K, label: "int|None" = None) -> Data:
    """
    Convert a single [R, R] connectivity matrix into a PyG Data graph.

    Node features: each node's row of the matrix (its connectivity profile).
    Edges: top_k strongest |weight| connections per node, symmetrized.
    """
    r = matrix.shape[0]
    x = torch.tensor(matrix, dtype=torch.float32)  # [R, R] node features

    edge_index_list = []
    edge_attr_list = []

    abs_matrix = np.abs(matrix)
    for i in range(r):
        # Indices of the top_k strongest edges from node i (excluding self).
        row = abs_matrix[i].copy()
        row[i] = -np.inf
        top_neighbors = np.argpartition(-row, min(top_k, r - 1) - 1)[:top_k]
        for j in top_neighbors:
            edge_index_list.append([i, int(j)])
            edge_attr_list.append(matrix[i, j])
            # Add the reverse direction too so the graph is undirected.
            edge_index_list.append([int(j), i])
            edge_attr_list.append(matrix[j, i])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).unsqueeze(-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data


def build_graph_sequence(
    matrices: np.ndarray, top_k: int = config.EDGE_TOP_K, label: "int|None" = None
) -> List[Data]:
    """
    Convert a sequence of matrices (e.g. pseudo-dynamic windows, shape
    [N_WINDOWS, R, R]) into a list of PyG Data graphs, one per window.
    """
    return [matrix_to_graph(matrices[w], top_k=top_k, label=label) for w in range(matrices.shape[0])]
