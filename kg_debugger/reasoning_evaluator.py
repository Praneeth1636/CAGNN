"""
Test multi-hop reasoning via path-based evaluation and reachability with curvature decay.
"""

import logging
import random
from typing import Dict, List, Any, Tuple

import networkx as nx
import pandas as pd

from .kg_loader import to_undirected

logger = logging.getLogger(__name__)


def _random_walk_path(U: nx.Graph, start: int, length: int, rng: random.Random) -> List[Tuple[int, str, int]]:
    """Random walk from start of given length. Returns list of (node, relation, prev_node)."""
    path = [start]
    rels = []
    cur = start
    for _ in range(length - 1):
        neighbors = list(U.neighbors(cur))
        if not neighbors:
            break
        nxt = rng.choice(neighbors)
        rel = U[cur][nxt].get("relation", "related_to")
        if "|" in str(rel):
            rel = str(rel).split("|")[0].strip()
        path.append(nxt)
        rels.append(rel)
        cur = nxt
    return list(zip(path[1:], rels, path[:-1]))


def generate_multi_hop_questions(
    G: nx.DiGraph,
    num_questions: int = 50,
    min_hops: int = 2,
    max_hops: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate multi-hop reasoning questions from the KG. Random walk 2--4 hops, create
    human-readable question requiring that path.

    Args:
        G: Directed KG.
        num_questions: Number of questions to generate.
        min_hops: Minimum path length.
        max_hops: Maximum path length.

    Returns:
        List of dicts: start, end, path (list of (node, relation, prev)), hops, question_text.
    """
    U = to_undirected(G)
    nodes = list(U.nodes())
    if len(nodes) < 3:
        return []
    rng = random.Random(42)
    questions = []
    attempts = 0
    while len(questions) < num_questions and attempts < num_questions * 5:
        attempts += 1
        start = rng.choice(nodes)
        hops = rng.randint(min_hops, max_hops)
        walk = _random_walk_path(U, start, hops + 1, rng)
        if len(walk) < min_hops:
            continue
        end = walk[-1][0]
        path_nodes = [start] + [w[0] for w in walk]
        path_rels = [w[1] for w in walk]
        question_text = f"What is the entity reachable from {G.nodes[start].get('label', start)} after following: " + " -> ".join(path_rels) + "?"
        questions.append({
            "start": start,
            "end": end,
            "path": [(p[0], p[1], p[2]) for p in walk],
            "hops": len(walk),
            "question_text": question_text,
        })
    return questions


def evaluate_reachability(
    G: nx.DiGraph,
    questions: List[Dict[str, Any]],
    edge_curvatures: Dict[Tuple[int, int], float],
    num_hops: int = 5,
) -> Tuple[float, List[float]]:
    """
    For each question, propagate signal from start for k hops using curvature as decay.
    More negative curvature = more signal loss. Return overall reachability score and per-question scores.

    Args:
        G: Directed KG.
        questions: List from generate_multi_hop_questions.
        edge_curvatures: Edge curvature (negative = decay).
        num_hops: Max propagation hops.

    Returns:
        (overall_score, list of per-question scores). Score in [0,1].
    """
    U = to_undirected(G)
    scores = []
    for q in questions:
        start = q["start"]
        end = q["end"]
        try:
            path = nx.shortest_path(U, start, end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            scores.append(0.0)
            continue
        if len(path) < 2:
            scores.append(1.0)
            continue
        # Signal strength: start at 1.0, multiply by (1 + curvature) per edge (clamped)
        signal = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            c = edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))
            decay = max(0.1, 1.0 + c)
            signal *= decay
        signal = max(0.0, min(1.0, signal))
        scores.append(signal)
    overall = sum(scores) / len(scores) if scores else 0.0
    return overall, scores


def compare_before_after(
    G_original: nx.DiGraph,
    G_rewired: nx.DiGraph,
    questions: List[Dict[str, Any]],
    edge_curvatures_original: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    """
    Run reachability on both graphs. Report per-question scores and how many improved.

    Args:
        G_original: Original KG.
        G_rewired: Rewired KG.
        questions: Same question list for both.
        edge_curvatures_original: Original curvature (for original reachability).

    Returns:
        DataFrame: question_id, start, end, score_before, score_after, improved.
    """
    from .kg_curvature import compute_kg_curvature
    try:
        _, edge_curv_after = compute_kg_curvature(G_rewired, alpha=0.5)
    except Exception:
        edge_curv_after = {}

    _, scores_before = evaluate_reachability(G_original, questions, edge_curvatures_original)
    _, scores_after = evaluate_reachability(G_rewired, questions, edge_curv_after)

    rows = []
    for i, q in enumerate(questions):
        sb = scores_before[i] if i < len(scores_before) else 0.0
        sa = scores_after[i] if i < len(scores_after) else 0.0
        rows.append({
            "question_id": i,
            "start": q["start"],
            "end": q["end"],
            "score_before": sb,
            "score_after": sa,
            "improved": sa > sb,
        })
    return pd.DataFrame(rows)
