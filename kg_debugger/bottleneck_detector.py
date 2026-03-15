"""
Detect reasoning bottlenecks: where multi-hop reasoning will fail.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .kg_loader import to_undirected

logger = logging.getLogger(__name__)


def detect_reasoning_bottlenecks(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    node_curvature: Dict[int, float],
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Find the top-k bottleneck edges (most negative curvature). For each: entities, types,
    relation, curvature, and impact score (edge betweenness — number of shortest paths through edge).

    Args:
        G: Directed KG.
        edge_curvatures: (u,v) -> curvature.
        node_curvature: node -> mean curvature (unused but kept for API).
        top_k: Number of top bottlenecks to return.

    Returns:
        DataFrame: source, target, source_label, target_label, source_type, target_type,
                   relation, curvature, impact_score. Sorted by impact (desc) then curvature (asc).
    """
    U = to_undirected(G)
    try:
        edge_bet = nx.edge_betweenness_centrality(U)
    except Exception:
        edge_bet = {e: 0.0 for e in U.edges()}

    rows = []
    seen = set()
    for u, v in U.edges():
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        c = edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))
        impact = edge_bet.get((u, v), edge_bet.get((v, u), 0.0))
        rel = U[u][v].get("relation", "unknown")
        if "|" in str(rel):
            rel = str(rel).split("|")[0].strip()
        rows.append({
            "source": u,
            "target": v,
            "source_label": G.nodes[u].get("label", str(u)),
            "target_label": G.nodes[v].get("label", str(v)),
            "source_type": G.nodes[u].get("type", "Entity"),
            "target_type": G.nodes[v].get("type", "Entity"),
            "relation": rel,
            "curvature": c,
            "impact_score": impact,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["impact_score", "curvature"], ascending=[False, True])
    return df.head(top_k).reset_index(drop=True)


def find_isolated_clusters(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    threshold: float = -0.2,
) -> List[Dict[str, Any]]:
    """
    Remove edges below curvature threshold and find disconnected components (knowledge islands).
    Report each cluster with its entities and the bottleneck edges connecting it.

    Args:
        G: Directed KG.
        edge_curvatures: Edge curvature dict.
        threshold: Edges with curvature < threshold are removed.

    Returns:
        List of dicts: {cluster_id, size, nodes, node_labels, bottleneck_edges_to_other}.
    """
    U = to_undirected(G)
    H = U.copy()
    H.remove_edges_from([
        (u, v) for u, v in U.edges()
        if edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0)) < threshold
    ])
    comps = list(nx.connected_components(H))
    # Bottleneck edges that connect different components in U
    bad_edges = [
        (u, v) for u, v in U.edges()
        if edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0)) < threshold
    ]
    comp_by_node = {}
    for i, c in enumerate(comps):
        for n in c:
            comp_by_node[n] = i

    result = []
    for i, c in enumerate(comps):
        node_labels = [U.nodes[n].get("label", str(n)) for n in c]
        bridge_edges = []
        for u, v in bad_edges:
            cu = comp_by_node.get(u)
            cv = comp_by_node.get(v)
            if cu == i and cv != i:
                bridge_edges.append((u, v, edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))))
            elif cv == i and cu != i:
                bridge_edges.append((v, u, edge_curvatures.get((v, u), edge_curvatures.get((u, v), 0.0))))
        result.append({
            "cluster_id": i,
            "size": len(c),
            "nodes": list(c),
            "node_labels": node_labels,
            "bottleneck_edges_to_other": bridge_edges,
        })
    return result


def trace_reasoning_path(
    G: nx.DiGraph,
    source_entity: int,
    target_entity: int,
    edge_curvatures: Dict[Tuple[int, int], float],
) -> List[Tuple[int, str, float]]:
    """
    Find shortest path from source to target. For each step return (entity_id, relation, curvature).
    If no path exists, return empty list and log.

    Args:
        G: Directed KG.
        source_entity: Start node id.
        target_entity: End node id.
        edge_curvatures: Edge curvature.

    Returns:
        List of (entity_id, relation, curvature) for each step along the path (entity is the
        node we step to; relation and curvature are for the edge we traversed). First element
        is (source_entity, "", 0.0).
    """
    U = to_undirected(G)
    try:
        path = nx.shortest_path(U, source_entity, target_entity)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        logger.info("No path between %s and %s", source_entity, target_entity)
        return []

    result = [(path[0], "", 0.0)]
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        rel = U[u][v].get("relation", "unknown")
        if "|" in str(rel):
            rel = str(rel).split("|")[0].strip()
        c = edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))
        result.append((v, rel, c))
    return result


def multi_hop_vulnerability_analysis(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    num_samples: int = 200,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Sample random entity pairs that are 3+ hops apart. For each, trace path and record
    minimum curvature along the path (weakest link). Return stats and DataFrame of vulnerable paths.

    Args:
        G: Directed KG.
        edge_curvatures: Edge curvature.
        num_samples: Number of pairs to sample.

    Returns:
        (stats_dict, df): stats has pct_paths_with_bottleneck, mean_min_curvature, etc.;
        df has columns start, end, path_length, min_curvature, is_vulnerable.
    """
    U = to_undirected(G)
    nodes = list(U.nodes())
    if len(nodes) < 2:
        return {"pct_paths_with_bottleneck": 0.0, "mean_min_curvature": 0.0}, pd.DataFrame()

    lengths = dict(nx.all_pairs_shortest_path_length(U))
    # Pairs that are 3+ hops apart
    long_pairs = []
    for _ in range(num_samples * 3):
        u, v = random.choice(nodes), random.choice(nodes)
        if u == v:
            continue
        d = lengths.get(u, {}).get(v, 999)
        if d >= 3:
            long_pairs.append((u, v))
            if len(long_pairs) >= num_samples:
                break
    if not long_pairs:
        return {"pct_paths_with_bottleneck": 0.0, "mean_min_curvature": 0.0}, pd.DataFrame()

    bottleneck_threshold = -0.1
    min_curvs = []
    vulnerable = 0
    rows = []
    for u, v in long_pairs:
        path = trace_reasoning_path(G, u, v, edge_curvatures)
        if len(path) < 2:
            continue
        curvatures_along = [path[i][2] for i in range(1, len(path))]
        min_c = min(curvatures_along)
        min_curvs.append(min_c)
        if min_c < bottleneck_threshold:
            vulnerable += 1
        rows.append({
            "start": u,
            "end": v,
            "path_length": len(path) - 1,
            "min_curvature": min_c,
            "is_vulnerable": min_c < bottleneck_threshold,
        })
    df = pd.DataFrame(rows)
    pct = (100.0 * vulnerable / len(rows)) if rows else 0.0
    mean_min = sum(min_curvs) / len(min_curvs) if min_curvs else 0.0
    stats = {
        "pct_paths_with_bottleneck": pct,
        "mean_min_curvature": mean_min,
        "num_sampled": len(rows),
        "num_vulnerable": vulnerable,
    }
    return stats, df


def generate_diagnostic_summary(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    node_curvature: Dict[int, float],
) -> Dict[str, Any]:
    """
    Run bottleneck detection, bridge entities, isolated clusters, and multi-hop vulnerability.
    Return a single structured dict with overall_health_score (0-100).

    Args:
        G: Directed KG.
        edge_curvatures: Edge curvature.
        node_curvature: Node mean curvature.

    Returns:
        Dict with: total_bottlenecks, worst_bottleneck_edges, bridge_entities, isolated_clusters,
        multi_hop_vulnerability_pct, overall_health_score.
    """
    from .kg_curvature import identify_bridge_entities

    bottleneck_threshold = -0.1
    total_bottlenecks = sum(1 for (u, v), c in edge_curvatures.items() if u < v and c < bottleneck_threshold)

    worst_df = detect_reasoning_bottlenecks(G, edge_curvatures, node_curvature, top_k=20)
    worst_bottleneck_edges = worst_df.to_dict("records") if not worst_df.empty else []

    bridge_entities = identify_bridge_entities(G, node_curvature, threshold=bottleneck_threshold)
    bridge_list = [{"node": x[0], "curvature": x[1], "betweenness": x[2], "label": x[3], "type": x[4]} for x in bridge_entities[:20]]

    isolated = find_isolated_clusters(G, edge_curvatures, threshold=-0.2)
    vulnerability_stats, _ = multi_hop_vulnerability_analysis(G, edge_curvatures, num_samples=min(200, G.number_of_nodes() * 2))
    multi_hop_vulnerability_pct = vulnerability_stats.get("pct_paths_with_bottleneck", 0.0)

    # Health score: 100 = no bottlenecks, 0 = all bad
    m = G.number_of_edges()
    if m == 0:
        health = 100.0
    else:
        pct_bad = 100.0 * total_bottlenecks / m
        health = max(0.0, 100.0 - pct_bad - multi_hop_vulnerability_pct * 0.5)
        health = min(100.0, health)

    return {
        "total_bottlenecks": total_bottlenecks,
        "worst_bottleneck_edges": worst_bottleneck_edges,
        "bridge_entities": bridge_list,
        "isolated_clusters": isolated,
        "multi_hop_vulnerability_pct": multi_hop_vulnerability_pct,
        "overall_health_score": round(health, 1),
    }
