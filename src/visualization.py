"""
Publication-quality plots and graph visualizations for curvature and over-squashing analysis.
All figures saved to figures/ at 150 DPI.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DPI = 150


def _ensure_fig_dir() -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return FIG_DIR


def plot_curvature_distribution(
    edge_curvatures: Dict[tuple, float],
    dataset_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel figure: (1) histogram of edge curvatures with red dashed line at zero,
    (2) bar chart of percentage of edges that are negative, near-zero, and positive.

    Args:
        edge_curvatures: Dict (u,v) -> curvature; use unique edges (e.g. u < v) to avoid double count.
        dataset_name: Used in title and filename.
        save_path: If None, saves to figures/curvature_distribution_{dataset_name}.png
    """
    vals = list(edge_curvatures.values())
    if not vals:
        logger.warning("No edge curvatures to plot")
        return
    # Deduplicate by (min,max) for undirected
    seen = set()
    unique_vals = []
    for (u, v), c in edge_curvatures.items():
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            unique_vals.append(c)
    vals = np.array(unique_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(vals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
    ax1.set_xlabel("Ollivier-Ricci curvature")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Edge curvature distribution ({dataset_name})")
    ax1.legend()

    neg = np.sum(vals < -0.05)
    near_zero = np.sum((vals >= -0.05) & (vals <= 0.05))
    pos = np.sum(vals > 0.05)
    total = len(vals)
    pct_neg = 100 * neg / total if total else 0
    pct_near = 100 * near_zero / total if total else 0
    pct_pos = 100 * pos / total if total else 0
    ax2.bar(["Negative\n(< -0.05)", "Near-zero\n[-0.05, 0.05]", "Positive\n(> 0.05)"],
            [pct_neg, pct_near, pct_pos], color=["#e74c3c", "#f1c40f", "#27ae60"], edgecolor="black")
    ax2.set_ylabel("Percentage of edges (%)")
    ax2.set_title(f"Curvature categories ({dataset_name})")

    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), f"curvature_distribution_{dataset_name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_curvature_vs_accuracy(
    df: pd.DataFrame,
    dataset_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel: (1) bar chart of accuracy per curvature bin, (2) overlapping histograms
    of curvature for correctly vs incorrectly classified nodes.

    Args:
        df: DataFrame from per_node_analysis (columns: curvature, correct).
        dataset_name: For title and filename.
        save_path: If None, saves to figures/curvature_vs_accuracy_{dataset_name}.png
    """
    if df.empty or "curvature" not in df.columns or "correct" not in df.columns:
        logger.warning("plot_curvature_vs_accuracy: empty df or missing columns")
        return
    df_clean = df.dropna(subset=["curvature"])
    if df_clean.empty:
        logger.warning("No valid curvature values for plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    n_bins = min(10, max(2, len(df_clean) // 20))
    df_clean = df_clean.copy()
    df_clean["bin"] = pd.qcut(df_clean["curvature"], q=n_bins, labels=False, duplicates="drop")
    acc_per_bin = df_clean.groupby("bin")["correct"].mean()
    bins_centers = np.arange(len(acc_per_bin))
    ax1.bar(bins_centers, acc_per_bin.values * 100, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Curvature bin")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Accuracy by curvature bin ({dataset_name})")

    correct_curv = df_clean.loc[df_clean["correct"], "curvature"]
    wrong_curv = df_clean.loc[~df_clean["correct"], "curvature"]
    ax2.hist([correct_curv, wrong_curv], bins=20, label=["Correct", "Incorrect"], alpha=0.7, color=["green", "red"])
    ax2.set_xlabel("Mean node curvature")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Curvature: correct vs incorrect ({dataset_name})")
    ax2.legend()

    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), f"curvature_vs_accuracy_{dataset_name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_graph_curvature(
    G_curved: nx.Graph,
    node_curvature: Dict[int, float],
    dataset_name: str,
    sample_size: int = 300,
    save_path: Optional[str] = None,
) -> None:
    """
    Spring layout visualization: nodes colored by mean curvature (RdYlGn colormap), edges with low alpha.

    Args:
        G_curved: Graph with curvature (can be large).
        node_curvature: node_id -> mean curvature.
        dataset_name: For title and filename.
        sample_size: If graph has more nodes, plot a random sample (and their induced subgraph).
        save_path: If None, saves to figures/graph_curvature_{dataset_name}.png
    """
    nodes = list(G_curved.nodes())
    if len(nodes) > sample_size:
        rng = np.random.default_rng(42)
        nodes = list(rng.choice(nodes, size=sample_size, replace=False))
        G_plot = G_curved.subgraph(nodes).copy()
    else:
        G_plot = G_curved

    pos = nx.spring_layout(G_plot, seed=42, k=0.5)
    curv_vals = [node_curvature.get(n, 0.0) for n in G_plot.nodes()]
    vmin = min(curv_vals) if curv_vals else 0
    vmax = max(curv_vals) if curv_vals else 1

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(G_plot, pos, alpha=0.2, ax=ax)
    nodes_draw = list(G_plot.nodes())
    sc = ax.scatter(
        [pos[n][0] for n in nodes_draw],
        [pos[n][1] for n in nodes_draw],
        c=[node_curvature.get(n, 0.0) for n in nodes_draw],
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        s=30,
        alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, label="Mean node curvature")
    ax.set_title(f"Graph curvature ({dataset_name})")
    ax.axis("off")
    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), f"graph_curvature_{dataset_name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_rewiring_comparison(
    results_df: pd.DataFrame,
    dataset_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart: original vs curvature-rewired vs random-rewired mean accuracy with error bars.

    Args:
        results_df: DataFrame with columns strategy, mean_test_acc, std_test_acc.
        dataset_name: For title and filename.
        save_path: If None, saves to figures/rewiring_comparison_{dataset_name}.png
    """
    if results_df.empty or "mean_test_acc" not in results_df.columns:
        logger.warning("plot_rewiring_comparison: empty or missing columns")
        return
    strategies = results_df["strategy"].tolist()
    means = results_df["mean_test_acc"].values * 100
    stds = results_df["std_test_acc"].values * 100 if "std_test_acc" in results_df.columns else np.zeros(len(means))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(strategies))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#3498db", "#27ae60", "#e74c3c"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Rewiring strategies comparison ({dataset_name})")
    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), f"rewiring_comparison_{dataset_name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_training_curves(
    histories: List[Dict[str, List[float]]],
    labels: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Line plot of test accuracy over epochs for multiple models.

    Args:
        histories: List of dicts with key "test_acc" (list of per-epoch values).
        labels: Model names for legend.
        save_path: If None, saves to figures/training_curves.png
    """
    if not histories or not labels or len(histories) != len(labels):
        logger.warning("plot_training_curves: need matching histories and labels")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, lab in zip(histories, labels):
        accs = h.get("test_acc", [])
        if not accs:
            continue
        ax.plot(np.arange(1, len(accs) + 1), np.array(accs) * 100, label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Training curves (test accuracy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), "training_curves.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_multiscale_hyperbolicity(
    multiscale_results: Dict[int, Dict[str, float]],
    dataset_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of correlation strength (curvature–accuracy) at each scale (radius).

    Args:
        multiscale_results: Dict scale -> {correlation, p_value, ...}.
        dataset_name: For title and filename.
        save_path: If None, saves to figures/multiscale_hyperbolicity_{dataset_name}.png
    """
    if not multiscale_results:
        logger.warning("No multiscale results to plot")
        return
    scales = sorted(multiscale_results.keys())
    corrs = [multiscale_results[s]["correlation"] for s in scales]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3498db" if c >= 0 else "#e74c3c" for c in corrs]
    ax.bar([str(s) for s in scales], corrs, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Scale (hop radius)")
    ax.set_ylabel("Correlation (hyperbolicity vs accuracy)")
    ax.set_title(f"Multiscale hyperbolicity–accuracy correlation ({dataset_name})")
    plt.tight_layout()
    path = save_path or os.path.join(_ensure_fig_dir(), f"multiscale_hyperbolicity_{dataset_name}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
