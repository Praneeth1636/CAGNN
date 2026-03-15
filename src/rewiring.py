"""
Curvature-based and random graph rewiring strategies to mitigate over-squashing.
"""

import logging
from typing import Dict, Set, Tuple

import torch
import networkx as nx
from torch_geometric.data import Data

from .utils import pyg_to_networkx

logger = logging.getLogger(__name__)


def curvature_based_rewiring(
    G: nx.Graph,
    edge_curvatures: Dict[Tuple[int, int], float],
    data: Data,
    threshold: float = -0.1,
    num_edges_add: int = 50,
) -> Data:
    """
    Rewire the graph by adding shortcut edges between bottleneck nodes (incident to
    highly negative curvature edges). For each such node, finds 2-hop neighbors that
    are also bottleneck-adjacent but not directly connected, and adds shortcut edges.

    Args:
        G: NetworkX graph used for curvature (e.g. LCC); used to compute 2-hop neighborhoods.
        edge_curvatures: Dict mapping (u, v) -> Ollivier-Ricci curvature.
        data: PyG Data object (full graph) to which new edges will be added.
        threshold: Edges with curvature below this are considered bottlenecks.
        num_edges_add: Maximum number of new (undirected) edges to add.

    Returns:
        New PyG Data object with same x, y, masks but modified edge_index (original + new edges).
    """
    # 1) Sort edges by curvature (most negative first)
    edges_below = [
        (u, v) for (u, v), c in edge_curvatures.items()
        if u < v and c < threshold  # each undirected edge once
    ]
    edges_below.sort(key=lambda uv: edge_curvatures[(uv[0], uv[1])])

    # 2) Bottleneck nodes: incident to at least one edge below threshold
    bottleneck: Set[int] = set()
    for u, v in edges_below:
        bottleneck.add(u)
        bottleneck.add(v)

    # Build adjacency set for G for fast lookups
    adj: Dict[int, Set[int]] = {n: set(G.neighbors(n)) for n in G.nodes()}

    # 3) For each bottleneck node, find 2-hop neighbors that are in bottleneck and not adjacent
    new_edges: Set[Tuple[int, int]] = set()
    for b in bottleneck:
        if len(new_edges) >= num_edges_add:
            break
        one_hop = adj.get(b, set())
        two_hop: Set[int] = set()
        for n in one_hop:
            two_hop.update(adj.get(n, set()))
        two_hop -= one_hop
        two_hop.discard(b)
        # Only add shortcuts to nodes that are also bottleneck and not already connected
        for w in two_hop:
            if w not in bottleneck:
                continue
            if w in one_hop:
                continue
            u, v = min(b, w), max(b, w)
            if (u, v) in new_edges:
                continue
            new_edges.add((u, v))
            if len(new_edges) >= num_edges_add:
                break

    # 4) Build new edge_index: original + new edges (both directions for undirected)
    edge_list = data.edge_index.t().tolist()
    existing = set((min(u, v), max(u, v)) for u, v in edge_list)
    added = 0
    for u, v in new_edges:
        if (u, v) in existing:
            continue
        edge_list.append([u, v])
        edge_list.append([v, u])
        existing.add((u, v))
        added += 1

    new_edge_index = torch.tensor(edge_list, dtype=data.edge_index.dtype, device=data.edge_index.device).t().contiguous()

    logger.info("Curvature-based rewiring: added %d new edges (target was up to %d).", added, num_edges_add)

    return Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


def random_rewiring(
    data: Data,
    num_edges_add: int = 50,
    seed: int = 42,
) -> Data:
    """
    Baseline: add random edges to the graph (same number as curvature-based rewiring).
    Used to show that curvature-guided rewiring outperforms random.

    Args:
        data: PyG Data object.
        num_edges_add: Number of new undirected edges to add.
        seed: Random seed for reproducibility.

    Returns:
        New PyG Data object with same attributes but edge_index including random new edges.
    """
    import random
    rng = random.Random(seed)
    n = data.num_nodes
    edge_set = set()
    for i in range(data.edge_index.size(1)):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        edge_set.add((min(u, v), max(u, v)))

    added = 0
    attempts = 0
    max_attempts = num_edges_add * 100
    while added < num_edges_add and attempts < max_attempts:
        attempts += 1
        u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if u == v:
            continue
        u, v = min(u, v), max(u, v)
        if (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        added += 1

    edge_list = list(edge_set)
    # PyG edge_index: both directions for undirected
    full_edges = []
    for u, v in edge_list:
        full_edges.append([u, v])
        full_edges.append([v, u])
    new_edge_index = torch.tensor(full_edges, dtype=data.edge_index.dtype, device=data.edge_index.device).t().contiguous()

    logger.info("Random rewiring: added %d new edges.", added)
    return Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
