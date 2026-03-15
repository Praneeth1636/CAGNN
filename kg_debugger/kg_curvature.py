"""
Apply curvature analysis to knowledge graphs. Wraps existing src.curvature.
"""

import logging
from typing import Dict, List, Tuple, Any

import networkx as nx
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curvature import compute_ollivier_ricci, get_node_curvature
from .kg_loader import to_undirected

logger = logging.getLogger(__name__)


def compute_kg_curvature(G: nx.DiGraph, alpha: float = 0.5) -> Tuple[nx.Graph, Dict[Tuple[int, int], float]]:
    """
    Compute Ollivier-Ricci curvature for the knowledge graph. Converts to undirected,
    runs on largest connected component, maps results back to original graph structure.

    Args:
        G: Directed knowledge graph (NetworkX DiGraph).
        alpha: Ollivier-Ricci alpha parameter (default 0.5).

    Returns:
        (G_curved, edge_curvatures): Undirected graph with curvature on edges,
        and dict (u,v) -> curvature. For edges not in LCC, curvature is 0.0.
    """
    U = to_undirected(G)
    try:
        G_curved, edge_curvatures = compute_ollivier_ricci(U, alpha=alpha)
    except Exception as e:
        logger.warning("Ollivier-Ricci failed (%s), using zeros", e)
        G_curved = U.copy()
        edge_curvatures = {}
        for u, v in U.edges():
            edge_curvatures[(u, v)] = 0.0
            edge_curvatures[(v, u)] = 0.0

    # Ensure all directed edges have a curvature (use undirected key)
    for u, v in G.edges():
        key = (u, v) if (u, v) in edge_curvatures else (v, u)
        if key not in edge_curvatures:
            edge_curvatures[(u, v)] = 0.0
        if (u, v) not in edge_curvatures:
            edge_curvatures[(u, v)] = edge_curvatures.get((v, u), 0.0)
        if (v, u) not in edge_curvatures:
            edge_curvatures[(v, u)] = edge_curvatures.get((u, v), 0.0)

    return G_curved, edge_curvatures


def compute_relation_curvature_stats(
    G_curved: nx.Graph,
    edge_curvatures: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    """
    Group curvature by relation type. For each relation type compute mean, std, min, max.
    Reveals which relation types are systematically weak (bottlenecks).

    Args:
        G_curved: Graph with edge curvature (undirected).
        edge_curvatures: Dict (u,v) -> curvature.

    Returns:
        DataFrame with columns: relation_type, mean_curvature, std_curvature, min_curvature, max_curvature, count.
    """
    by_rel: Dict[str, List[float]] = {}
    seen = set()
    for u, v in G_curved.edges():
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        c = edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))
        rel = G_curved[u][v].get("relation", "unknown")
        if "|" in str(rel):
            for r in str(rel).split("|"):
                r = r.strip() or "unknown"
                by_rel.setdefault(r, []).append(c)
        else:
            by_rel.setdefault(rel, []).append(c)

    rows = []
    for rel, curv_list in by_rel.items():
        if not curv_list:
            continue
        import numpy as np
        arr = np.array(curv_list)
        rows.append({
            "relation_type": rel,
            "mean_curvature": float(np.mean(arr)),
            "std_curvature": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "min_curvature": float(np.min(arr)),
            "max_curvature": float(np.max(arr)),
            "count": len(arr),
        })
    return pd.DataFrame(rows).sort_values("mean_curvature")


def compute_entity_type_curvature(
    G_curved: nx.Graph,
    node_curvature: Dict[int, float],
) -> pd.DataFrame:
    """
    Group node curvature by entity type. Shows which entity types sit in bottleneck regions.

    Args:
        G_curved: Graph with node attributes (type).
        node_curvature: Dict node_id -> mean curvature.

    Returns:
        DataFrame: entity_type, mean_curvature, std_curvature, count.
    """
    by_type: Dict[str, List[float]] = {}
    for n in G_curved.nodes():
        t = G_curved.nodes[n].get("type", "Entity")
        c = node_curvature.get(n, 0.0)
        by_type.setdefault(t, []).append(c)
    import numpy as np
    rows = []
    for t, curv_list in by_type.items():
        arr = np.array(curv_list)
        rows.append({
            "entity_type": t,
            "mean_curvature": float(np.mean(arr)),
            "std_curvature": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "count": len(arr),
        })
    return pd.DataFrame(rows).sort_values("mean_curvature")


def identify_bridge_entities(
    G: nx.DiGraph,
    node_curvature: Dict[int, float],
    threshold: float = -0.1,
) -> List[Tuple[int, float, float, str, str]]:
    """
    Find entities that are critical bridges: negative mean curvature AND high betweenness.
    These are single points of failure in the KG.

    Args:
        G: Directed KG (we use undirected for betweenness).
        node_curvature: Node id -> mean curvature.
        threshold: Curvature below this counts as "bottleneck region".

    Returns:
        List of (node_id, curvature, betweenness, label, type), sorted by betweenness (desc).
    """
    U = to_undirected(G)
    betweenness = nx.edge_betweenness_centrality(U)
    node_betweenness: Dict[int, float] = {n: 0.0 for n in U.nodes()}
    for (u, v), b in betweenness.items():
        node_betweenness[u] += b
        node_betweenness[v] += b
    # Normalize by number of edges
    m = U.number_of_edges()
    if m > 0:
        for n in node_betweenness:
            node_betweenness[n] /= m

    candidates = []
    for n in U.nodes():
        c = node_curvature.get(n, 0.0)
        if c >= threshold:
            continue
        b = node_betweenness.get(n, 0.0)
        label = U.nodes[n].get("label", str(n))
        etype = U.nodes[n].get("type", "Entity")
        candidates.append((n, c, b, label, etype))
    candidates.sort(key=lambda x: -x[2])
    return candidates
