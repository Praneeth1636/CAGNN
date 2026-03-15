"""
Ollivier-Ricci curvature and local Gromov hyperbolicity computation.
"""

# Avoid OpenMP crash on macOS when multiple libomp copies are loaded (NetworKit, PyTorch, NumPy).
# Must be set before any library that uses OpenMP is imported.
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import logging
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional import for Ollivier-Ricci; fail gracefully if not installed
try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    HAS_GRAPH_RICCI = True
except ImportError:
    HAS_GRAPH_RICCI = False
    OllivierRicci = None  # type: ignore


def _largest_connected_component_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as an induced subgraph (preserving node labels)."""
    if nx.is_connected(G):
        return G
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def compute_ollivier_ricci(
    G: nx.Graph,
    alpha: float = 0.5,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], float]]:
    """
    Compute Ollivier-Ricci curvature for all edges using GraphRicciCurvature.
    For disconnected graphs, uses the largest connected component.

    Args:
        G: Undirected NetworkX graph.
        alpha: Parameter for the Ollivier-Ricci computation (default 0.5).

    Returns:
        (G_curved, edge_curvatures): The graph with curvature edge attributes (on LCC),
        and a dict mapping (u, v) edge tuple to curvature value. For undirected edges
        both (u,v) and (v,u) are stored with the same value.
    """
    if not HAS_GRAPH_RICCI or OllivierRicci is None:
        raise ImportError("GraphRicciCurvature is required. Install with: pip install GraphRicciCurvature>=0.5.3")

    G_work = _largest_connected_component_subgraph(G)
    orc = OllivierRicci(G_work, alpha=alpha, verbose="ERROR")
    orc.compute_ricci_curvature()
    G_curved = orc.G.copy()

    edge_curvatures: Dict[Tuple[int, int], float] = {}
    for u, v in G_curved.edges():
        curv = G_curved[u][v].get("ricciCurvature", float("nan"))
        if np.isnan(curv):
            curv = 0.0  # fallback for any edge missing curvature
        edge_curvatures[(u, v)] = float(curv)
        edge_curvatures[(v, u)] = float(curv)

    logger.info("Computed Ollivier-Ricci curvature on %d edges (LCC has %d nodes).",
                len(edge_curvatures) // 2, G_curved.number_of_nodes())
    return G_curved, edge_curvatures


def get_node_curvature(G_curved: nx.Graph) -> Dict[int, float]:
    """
    Compute mean curvature per node from its incident edges.
    Expects G_curved to have edge attribute 'ricciCurvature' (or use edge_curvatures to build G_curved).

    Args:
        G_curved: NetworkX graph with edge curvature (e.g. from compute_ollivier_ricci).
                 Edges should have 'ricciCurvature' or we use 0.0.

    Returns:
        Dict mapping node id -> mean curvature over incident edges.
    """
    node_curvature: Dict[int, float] = {}
    for n in G_curved.nodes():
        curvatures = []
        for _, _, d in G_curved.edges(n, data=True):
            c = d.get("ricciCurvature", 0.0)
            if hasattr(c, "item"):
                c = float(c)
            curvatures.append(c)
        node_curvature[n] = float(np.mean(curvatures)) if curvatures else 0.0
    return node_curvature


def _bfs_subgraph(G: nx.Graph, source: int, radius: int) -> nx.Graph:
    """Return the subgraph induced by nodes within `radius` hops from `source`."""
    if radius <= 0:
        return G.subgraph([source]).copy()
    nodes = {source}
    current = {source}
    for _ in range(radius):
        next_layer = set()
        for u in current:
            next_layer.update(G.neighbors(u))
        current = next_layer - nodes
        nodes |= current
        if not current:
            break
    return G.subgraph(nodes).copy()


def local_gromov_hyperbolicity(
    G: nx.Graph,
    node: int,
    radius: int = 2,
    max_quadruples: int = 500,
    seed: Optional[int] = None,
) -> float:
    """
    Compute Gromov delta-hyperbolicity on the k-hop neighborhood of a node.
    Samples up to max_quadruples quadruples (x,y,z,w) and computes
    delta = (max_sum - second_max_sum) / 2 for the three distance sums
    d(x,y)+d(z,w), d(x,z)+d(y,w), d(x,w)+d(y,z). Returns the maximum delta.

    Args:
        G: Undirected NetworkX graph.
        node: Center node for the neighborhood.
        radius: Hop radius for the neighborhood subgraph.
        max_quadruples: Maximum number of quadruples to sample.
        seed: Random seed for sampling quadruples.

    Returns:
        Maximum delta (float) over the sampled quadruples; 0.0 if fewer than 4 nodes.
    """
    rng = random.Random(seed)
    H = _bfs_subgraph(G, node, radius)
    nodes = list(H.nodes())
    if len(nodes) < 4:
        return 0.0

    try:
        lengths = dict(nx.all_pairs_shortest_path_length(H))
    except Exception:
        return 0.0

    def d(a: int, b: int) -> float:
        return float(lengths.get(a, {}).get(b, float("inf")))

    # Sample quadruples
    if len(nodes) <= 20:
        quads = list(itertools.combinations(nodes, 4))
        if len(quads) > max_quadruples:
            quads = rng.sample(quads, max_quadruples)
    else:
        quads = []
        for _ in range(max_quadruples):
            quad = tuple(rng.sample(nodes, 4))
            quads.append(quad)

    max_delta = 0.0
    for (x, y, z, w) in quads:
        s1 = d(x, y) + d(z, w)
        s2 = d(x, z) + d(y, w)
        s3 = d(x, w) + d(y, z)
        if np.isinf(s1) or np.isinf(s2) or np.isinf(s3):
            continue
        three = sorted([s1, s2, s3], reverse=True)
        delta = (three[0] - three[1]) / 2.0
        max_delta = max(max_delta, delta)

    return max_delta


def compute_local_hyperbolicity_all(
    G: nx.Graph,
    radius: int = 2,
    max_quadruples: int = 500,
    seed: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute local Gromov hyperbolicity for every node in the graph.
    Uses tqdm for progress.

    Args:
        G: Undirected NetworkX graph.
        radius: Hop radius for each neighborhood.
        max_quadruples: Max quadruples per node.
        seed: Random seed.

    Returns:
        Dict mapping node -> local hyperbolicity (max delta).
    """
    result: Dict[int, float] = {}
    nodes = list(G.nodes())
    for n in tqdm(nodes, desc="Local hyperbolicity", unit="node"):
        result[n] = local_gromov_hyperbolicity(G, n, radius=radius, max_quadruples=max_quadruples, seed=seed)
    return result
