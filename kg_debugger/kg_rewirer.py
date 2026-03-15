"""
Suggest and apply fixes to weak connections in the KG. Uses curvature-based ideas from src.rewiring.
"""

import logging
from typing import Dict, List, Tuple, Any

import networkx as nx
import pandas as pd

from .kg_loader import to_undirected

logger = logging.getLogger(__name__)


def suggest_new_connections(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    node_curvature: Dict[int, float],
    max_suggestions: int = 20,
) -> pd.DataFrame:
    """
    For each bottleneck region, suggest new edges that would reduce over-squashing.
    Each suggestion: source, target, suggested_relation, expected_improvement, priority.

    Args:
        G: Directed KG.
        edge_curvatures: Edge curvature.
        node_curvature: Node mean curvature.
        max_suggestions: Maximum number of suggestions.

    Returns:
        DataFrame: source, target, source_label, target_label, suggested_relation,
                    expected_improvement, priority (high/medium/low).
    """
    U = to_undirected(G)
    threshold = -0.1
    bottleneck_edges = [(u, v) for (u, v), c in edge_curvatures.items() if u < v and c < threshold]
    bottleneck_edges.sort(key=lambda uv: edge_curvatures[(uv[0], uv[1])])
    bottleneck_nodes = set()
    for u, v in bottleneck_edges:
        bottleneck_nodes.add(u)
        bottleneck_nodes.add(v)

    adj = {n: set(U.neighbors(n)) for n in U.nodes()}
    suggestions = []
    seen = set()
    try:
        edge_bet = nx.edge_betweenness_centrality(U)
    except Exception:
        edge_bet = {}

    for b in bottleneck_nodes:
        if len(suggestions) >= max_suggestions:
            break
        one_hop = adj.get(b, set())
        two_hop = set()
        for n in one_hop:
            two_hop.update(adj.get(n, set()))
        two_hop -= one_hop
        two_hop.discard(b)
        for w in two_hop:
            if w not in bottleneck_nodes:
                continue
            if w in one_hop:
                continue
            key = (min(b, w), max(b, w))
            if key in seen:
                continue
            seen.add(key)
            impact_b = edge_bet.get((b, list(one_hop)[0]) if one_hop else (b, w), 0.0)
            impact_w = edge_bet.get((w, list(adj.get(w, set()))[0]) if adj.get(w) else (w, b), 0.0)
            impact = (impact_b + impact_w) / 2
            curv_b = node_curvature.get(b, 0.0)
            curv_w = node_curvature.get(w, 0.0)
            expected_improvement = -min(curv_b, curv_w)
            rel = "collaborates_with"
            for _, _, d in G.edges(b, data=True):
                rel = d.get("relation", rel)
                break
            priority = "high" if impact > 0.01 and expected_improvement > 0.1 else ("medium" if impact > 0.005 else "low")
            suggestions.append({
                "source": b,
                "target": w,
                "source_label": G.nodes[b].get("label", str(b)),
                "target_label": G.nodes[w].get("label", str(w)),
                "suggested_relation": rel,
                "expected_improvement": round(expected_improvement, 4),
                "priority": priority,
            })
            if len(suggestions) >= max_suggestions:
                break

    return pd.DataFrame(suggestions)


def apply_suggestions(G: nx.DiGraph, suggestions_df: pd.DataFrame) -> nx.DiGraph:
    """
    Add the suggested edges to the graph. Returns a new graph.

    Args:
        G: Original directed KG.
        suggestions_df: DataFrame from suggest_new_connections (columns source, target, suggested_relation).

    Returns:
        New NetworkX DiGraph with added edges.
    """
    H = G.copy()
    if suggestions_df.empty or "source" not in suggestions_df.columns:
        return H
    added = 0
    for _, row in suggestions_df.iterrows():
        u, v = int(row["source"]), int(row["target"])
        rel = row.get("suggested_relation", "related_to")
        if u == v:
            continue
        if not H.has_edge(u, v):
            H.add_edge(u, v, relation=rel, weight=1.0)
            added += 1
        if not H.has_edge(v, u):
            H.add_edge(v, u, relation=rel, weight=1.0)
            added += 1
    logger.info("Applied %d new edges from suggestions", added)
    return H


def evaluate_rewiring_impact(
    G_original: nx.DiGraph,
    G_rewired: nx.DiGraph,
    edge_curvatures_original: Dict[Tuple[int, int], float],
) -> Dict[str, Any]:
    """
    Compare before/after: recompute curvature on rewired graph, compare bottleneck count
    and multi-hop vulnerability. Return before/after comparison dict.

    Args:
        G_original: Original KG.
        G_rewired: KG after applying suggestions.
        edge_curvatures_original: Original edge curvature (for before stats).

    Returns:
        Dict: before_bottlenecks, after_bottlenecks, before_vulnerability_pct, after_vulnerability_pct,
        health_before, health_after, improvement.
    """
    from .kg_curvature import compute_kg_curvature
    from .bottleneck_detector import multi_hop_vulnerability_analysis

    threshold = -0.1
    before_bottlenecks = sum(1 for (u, v), c in edge_curvatures_original.items() if u < v and c < threshold)
    m_orig = G_original.number_of_edges()
    health_before = max(0, 100 - 100 * before_bottlenecks / m_orig) if m_orig else 100

    try:
        _, edge_curv_after = compute_kg_curvature(G_rewired, alpha=0.5)
    except Exception:
        edge_curv_after = {}
    after_bottlenecks = sum(1 for (u, v), c in edge_curv_after.items() if u < v and c < threshold)
    m_rew = G_rewired.number_of_edges()
    health_after = max(0, 100 - 100 * after_bottlenecks / m_rew) if m_rew else 100

    vul_before, _ = multi_hop_vulnerability_analysis(G_original, edge_curvatures_original, num_samples=min(100, G_original.number_of_nodes()))
    vul_after, _ = multi_hop_vulnerability_analysis(G_rewired, edge_curv_after, num_samples=min(100, G_rewired.number_of_nodes()))
    before_vul = vul_before.get("pct_paths_with_bottleneck", 0.0)
    after_vul = vul_after.get("pct_paths_with_bottleneck", 0.0)

    improvement = (health_after - health_before) + (before_vul - after_vul) * 0.5

    return {
        "before_bottlenecks": before_bottlenecks,
        "after_bottlenecks": after_bottlenecks,
        "before_vulnerability_pct": before_vul,
        "after_vulnerability_pct": after_vul,
        "health_before": round(health_before, 1),
        "health_after": round(health_after, 1),
        "improvement": round(improvement, 2),
    }


def auto_fix(
    G: nx.DiGraph,
    edge_curvatures: Dict[Tuple[int, int], float],
    node_curvature: Dict[int, float],
    budget: int = 20,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Fully automatic: detect bottlenecks, generate suggestions, apply top-`budget` suggestions,
    evaluate impact. Return fixed graph and improvement report.

    Args:
        G: Directed KG.
        edge_curvatures: Edge curvature.
        node_curvature: Node curvature.
        budget: Max number of new edges to add.

    Returns:
        (G_fixed, report_dict).
    """
    suggestions_df = suggest_new_connections(G, edge_curvatures, node_curvature, max_suggestions=budget)
    if suggestions_df.empty:
        return G.copy(), {"message": "No suggestions generated", "improvement": 0.0}
    G_fixed = apply_suggestions(G, suggestions_df)
    report = evaluate_rewiring_impact(G, G_fixed, edge_curvatures)
    report["num_edges_added"] = G_fixed.number_of_edges() - G.number_of_edges()
    return G_fixed, report
