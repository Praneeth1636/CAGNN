"""
Data loading, graph conversion, statistics, and configuration for curvature-aware GNN analysis.
"""

import logging
from typing import Any, Dict

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: all hyperparameters in one place (no magic numbers in code)
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "learning_rate": 0.01,
    "hidden_dim": 64,
    "dropout": 0.5,
    "epochs": 200,
    "num_gnn_layers": 2,
    "curvature_alpha": 0.5,
    "rewiring_threshold": -0.1,
    "num_edges_add": 50,
    "gat_heads": 8,
    "gat_hidden_per_head": 8,
    "gat_dropout": 0.6,
}


def load_dataset(name: str = "Cora", root: str = "./data") -> Data:
    """
    Load a citation network dataset (Cora, CiteSeer, or PubMed) from PyTorch Geometric.

    Args:
        name: One of "Cora", "CiteSeer", "PubMed".
        root: Root directory where the dataset will be stored.

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.

    Raises:
        ValueError: If dataset name is not supported.
    """
    name = name.lower()
    if name not in ("cora", "citeseer", "pubmed"):
        raise ValueError(f"Dataset must be one of Cora, CiteSeer, PubMed; got '{name}'")
    name_cap = name.capitalize()
    dataset = Planetoid(root=root, name=name_cap)
    data = dataset[0]
    logger.info("Loaded %s: %d nodes, %d edges, %d features, %d classes",
                name_cap, data.num_nodes, data.edge_index.size(1), data.num_node_features, dataset.num_classes)
    return data


def pyg_to_networkx(data: Data) -> nx.Graph:
    """
    Convert a PyTorch Geometric Data object to an undirected NetworkX graph.

    Node features and labels are not copied; only topology (nodes and edges) is preserved.
    Node indices in the NetworkX graph match the PyG node indices.

    Args:
        data: PyG Data with edge_index of shape [2, num_edges].

    Returns:
        Undirected NetworkX Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # Avoid duplicate edges for undirected: add each (u,v) once; NetworkX will treat as undirected
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            G.add_edge(u, v)
    return G


def get_graph_stats(G: nx.Graph) -> None:
    """
    Print summary statistics of a graph: nodes, edges, average degree, approximate diameter,
    average clustering coefficient, and number of connected components.

    For disconnected graphs, diameter is computed on the largest connected component.
    For large graphs, diameter may be approximated or expensive; we use nx.diameter
    on the largest component (may be slow for very large graphs).

    Args:
        G: NetworkX graph (can be directed or undirected).
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0.0

    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        num_components = len(comps)
        largest = max(comps, key=len)
        G_lcc = G.subgraph(largest).copy()
        try:
            diameter = nx.diameter(G_lcc)
        except (nx.NetworkXError, nx.NetworkXPointlessConcept):
            diameter = float("nan")
        logger.info(
            "Graph stats: nodes=%d, edges=%d, avg_degree=%.2f, diameter(LCC)=%s, "
            "avg_clustering=%.4f, num_components=%d",
            num_nodes, num_edges, avg_degree, diameter if not (diameter != diameter) else "N/A",
            nx.average_clustering(G), num_components,
        )
    else:
        try:
            diameter = nx.diameter(G)
        except (nx.NetworkXError, nx.NetworkXPointlessConcept):
            diameter = float("nan")
        logger.info(
            "Graph stats: nodes=%d, edges=%d, avg_degree=%.2f, diameter=%s, avg_clustering=%.4f, num_components=1",
            num_nodes, num_edges, avg_degree, diameter if not (diameter != diameter) else "N/A",
            nx.average_clustering(G),
        )
