"""
KG-specific visualizations. All figures saved to figures/ with kg_ prefix at 150 DPI.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .kg_loader import to_undirected

logger = logging.getLogger(__name__)
DPI = 150
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _node_color_by_type(G: nx.Graph) -> List[str]:
    cmap = {"Person": "#3498db", "Team": "#2ecc71", "Project": "#9b59b6", "Technology": "#e74c3c",
            "Location": "#f39c12", "Document": "#1abc9c", "Entity": "#95a5a6", "Synset": "#34495e"}
    return [cmap.get(G.nodes[n].get("type", "Entity"), "#95a5a6") for n in G.nodes()]


def plot_kg_overview(
    G: nx.DiGraph,
    node_curvature: Dict[int, float],
    title: str = "Knowledge Graph Overview",
    edge_curvatures: Optional[Dict[Tuple[int, int], float]] = None,
) -> str:
    """
    Full KG: nodes colored by entity type, sized by degree. Edges colored by curvature (red=bottleneck, green=healthy).
    Uses spring layout. Saves to figures/kg_overview.png.

    Args:
        G: Directed KG.
        node_curvature: Node mean curvature.
        title: Plot title.
        edge_curvatures: Optional dict (u,v)->curvature for edge coloring; else derived from node curvature.

    Returns:
        Path to saved figure.
    """
    U = to_undirected(G)
    if U.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(U, seed=42, k=2.0 / np.sqrt(U.number_of_nodes()))
    node_colors = _node_color_by_type(U)
    deg = dict(U.degree())
    node_sizes = [100 + 50 * deg[n] for n in U.nodes()]
    if edge_curvatures:
        edge_curv = {}
        for u, v in U.edges():
            edge_curv[(u, v)] = edge_curvatures.get((u, v), edge_curvatures.get((v, u), 0.0))
    else:
        edge_curv = {}
        for u, v in U.edges():
            edge_curv[(u, v)] = (node_curvature.get(u, 0) + node_curvature.get(v, 0)) / 2
    edge_colors = [edge_curv.get((u, v), 0) for u, v in U.edges()]
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(U, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(U, pos, edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn, edge_vmin=-0.3, edge_vmax=0.3, ax=ax, width=0.5)
    ax.set_title(title)
    ax.axis("off")
    path = FIG_DIR / "kg_overview.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def plot_bottleneck_subgraph(
    G: nx.DiGraph,
    bottleneck_edges: List[Tuple[int, int]],
    edge_curvatures: Dict[Tuple[int, int], float],
    title: str = "Bottleneck Region",
) -> str:
    """
    Zoom into bottleneck region: bottleneck edges in red, others grey. Label entities.
    Saves to figures/kg_bottleneck_subgraph.png.

    Args:
        G: Directed KG.
        bottleneck_edges: List of (u,v) bottleneck edges.
        edge_curvatures: Edge curvature.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    if not bottleneck_edges:
        return ""
    nodes = set()
    for u, v in bottleneck_edges:
        nodes.add(u)
        nodes.add(v)
    U = to_undirected(G)
    sub = U.subgraph(nodes).copy()
    if sub.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(sub, seed=42)
    edge_colors = []
    for u, v in sub.edges():
        key = (min(u, v), max(u, v))
        is_bottleneck = (u, v) in bottleneck_edges or (v, u) in bottleneck_edges
        edge_colors.append("red" if is_bottleneck else "lightgray")
    labels = {n: sub.nodes[n].get("label", str(n))[:12] for n in sub.nodes()}
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(sub, pos, node_color="#3498db", ax=ax)
    nx.draw_networkx_edges(sub, pos, edge_color=edge_colors, ax=ax)
    nx.draw_networkx_labels(sub, pos, labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    path = FIG_DIR / "kg_bottleneck_subgraph.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def plot_reasoning_path(
    G: nx.DiGraph,
    path_data: List[Tuple[int, str, float]],
    title: str = "Reasoning Path",
) -> str:
    """
    Visualize a multi-hop path. Entities as labeled nodes, edges colored by curvature. Highlight weakest link.
    path_data: list of (entity_id, relation, curvature) from trace_reasoning_path.

    Args:
        G: Directed KG.
        path_data: Output of trace_reasoning_path.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    if len(path_data) < 2:
        return ""
    U = to_undirected(G)
    nodes = [path_data[i][0] for i in range(len(path_data))]
    sub = U.subgraph(nodes).copy()
    pos = nx.spring_layout(sub, seed=42)
    edge_colors = []
    for i in range(1, len(path_data)):
        edge_colors.append(path_data[i][2])
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(sub, pos, node_color="#3498db", ax=ax)
    for i in range(1, len(path_data)):
        u, v = path_data[i - 1][0], path_data[i][0]
        c = path_data[i][2]
        color = cmap((c + 0.5) / 0.6) if c >= -0.3 else (1, 0, 0, 1)
        nx.draw_networkx_edges(sub, pos, [(u, v)], edge_color=[color], width=3, ax=ax)
    labels = {n: sub.nodes[n].get("label", str(n))[:15] for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels, font_size=9, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    path = FIG_DIR / "kg_reasoning_path.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def plot_before_after_comparison(
    G_original: nx.DiGraph,
    G_rewired: nx.DiGraph,
    node_curv_original: Dict[int, float],
    node_curv_rewired: Dict[int, float],
) -> str:
    """
    Side-by-side: original (bottlenecks) vs rewired (improvements). Same layout.

    Args:
        G_original: Original KG.
        G_rewired: Rewired KG.
        node_curv_original: Node curvature before.
        node_curv_rewired: Node curvature after.

    Returns:
        Path to saved figure.
    """
    U1 = to_undirected(G_original)
    U2 = to_undirected(G_rewired)
    if U1.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(U1, seed=42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors1 = [node_curv_original.get(n, 0) for n in U1.nodes()]
    colors2 = [node_curv_rewired.get(n, 0) for n in U2.nodes()] if U2.number_of_nodes() == U1.number_of_nodes() else [0] * U1.number_of_nodes()
    nx.draw_networkx_nodes(U1, pos, node_color=colors1, cmap=plt.cm.RdYlGn, node_size=80, vmin=-0.3, vmax=0.3, ax=ax1)
    nx.draw_networkx_edges(U1, pos, ax=ax1, alpha=0.5)
    ax1.set_title("Before (curvature)")
    ax1.axis("off")
    nx.draw_networkx_nodes(U2, pos, node_color=colors2, cmap=plt.cm.RdYlGn, node_size=80, vmin=-0.3, vmax=0.3, ax=ax2)
    nx.draw_networkx_edges(U2, pos, ax=ax2, alpha=0.5)
    ax2.set_title("After (curvature)")
    ax2.axis("off")
    path = FIG_DIR / "kg_before_after.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def plot_relation_curvature_heatmap(relation_stats_df: pd.DataFrame) -> str:
    """
    Heatmap: mean curvature per relation type. Red = weak, green = strong.

    Args:
        relation_stats_df: From compute_relation_curvature_stats.

    Returns:
        Path to saved figure.
    """
    if relation_stats_df.empty:
        return ""
    df = relation_stats_df.sort_values("mean_curvature").head(20)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))
    vals = df["mean_curvature"].values.reshape(-1, 1)
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["relation_type"].tolist(), fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean curvature")
    ax.set_title("Relation type curvature")
    path = FIG_DIR / "kg_relation_heatmap.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def plot_health_dashboard(diagnostic_summary: Dict[str, Any]) -> str:
    """
    Single-figure dashboard: health score gauge, bottleneck count bar, curvature distribution, multi-hop vulnerability pie.

    Args:
        diagnostic_summary: From generate_diagnostic_summary.

    Returns:
        Path to saved figure.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    health = diagnostic_summary.get("overall_health_score", 0)
    ax1.barh([0], [health], color="green" if health >= 70 else "orange" if health >= 40 else "red", height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Health score")
    ax1.set_title("Overall health")
    ax1.set_yticks([])

    nb = diagnostic_summary.get("total_bottlenecks", 0)
    ax2.bar(["Bottlenecks"], [nb], color="coral")
    ax2.set_title("Bottleneck count")
    ax2.set_ylabel("Count")

    curvatures = [e.get("curvature", 0) for e in diagnostic_summary.get("worst_bottleneck_edges", [])]
    if curvatures:
        ax3.hist(curvatures, bins=15, color="steelblue", edgecolor="black")
    ax3.set_xlabel("Curvature")
    ax3.set_title("Bottleneck curvature distribution")

    vul = diagnostic_summary.get("multi_hop_vulnerability_pct", 0)
    ax4.pie([vul, 100 - vul], labels=["Vulnerable", "OK"], autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"])
    ax4.set_title("Multi-hop vulnerability")

    plt.tight_layout()
    path = FIG_DIR / "kg_health_dashboard.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return str(path)
