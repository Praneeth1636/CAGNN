"""
Over-squashing detection and curvature-accuracy correlation analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.data import Data

from .curvature import compute_local_hyperbolicity_all
from .models import evaluate_model, train_model
from .rewiring import curvature_based_rewiring, random_rewiring
from .utils import CONFIG, pyg_to_networkx

logger = logging.getLogger(__name__)


def per_node_analysis(
    model: torch.nn.Module,
    data: Data,
    node_curvature: Dict[int, float],
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    For each test node, record node_id, curvature, correct (bool), predicted label, true label.
    Nodes not in node_curvature get NaN curvature.

    Args:
        model: Trained GNN model.
        data: PyG Data with test_mask and y.
        node_curvature: Dict node_id -> mean curvature.
        device: Device for model/data.

    Returns:
        DataFrame with columns: node_id, curvature, correct, pred_label, true_label.
    """
    if device is None:
        device = next(model.parameters()).device
    _, _, _, pred = evaluate_model(model, data, device)
    pred = pred.cpu().numpy()
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    rows = []
    for i in np.where(test_mask)[0]:
        curvature = node_curvature.get(int(i), float("nan"))
        correct = bool(pred[i] == y[i])
        rows.append({
            "node_id": int(i),
            "curvature": curvature,
            "correct": correct,
            "pred_label": int(pred[i]),
            "true_label": int(y[i]),
        })
    return pd.DataFrame(rows)


def curvature_accuracy_correlation(df: pd.DataFrame) -> float:
    """
    Bin nodes by curvature into 10 bins, compute accuracy per bin, and Pearson correlation
    between curvature and correctness. Prints results.

    Args:
        df: DataFrame from per_node_analysis with columns curvature, correct.

    Returns:
        Pearson correlation coefficient between curvature and correct (0/1).
    """
    if df.empty or "curvature" not in df.columns or "correct" not in df.columns:
        logger.warning("curvature_accuracy_correlation: empty df or missing columns")
        return 0.0

    df_clean = df.dropna(subset=["curvature"])
    if len(df_clean) < 10:
        logger.warning("Too few nodes with curvature for correlation")
        return 0.0

    # Bins
    df_clean = df_clean.copy()
    df_clean["curvature_bin"] = pd.qcut(df_clean["curvature"], q=10, labels=False, duplicates="drop")
    acc_per_bin = df_clean.groupby("curvature_bin")["correct"].mean()
    logger.info("Accuracy per curvature bin:\n%s", acc_per_bin.to_string())

    r, p = stats.pearsonr(df_clean["curvature"], df_clean["correct"].astype(float))
    logger.info("Pearson correlation (curvature vs correct): r=%.4f, p=%.4e", r, p)
    return float(r)


def multiscale_analysis(
    G: Any,
    data: Data,
    model: torch.nn.Module,
    scales: List[int],
    device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, float]]:
    """
    At each scale (radius), compute local hyperbolicity for all nodes, then correlate
    with prediction correctness. Uses scipy.stats.pearsonr.

    Args:
        G: NetworkX graph (e.g. from pyg_to_networkx(data)).
        data: PyG Data with test_mask and y.
        model: Trained GNN (for predictions).
        scales: List of hop radii for local hyperbolicity.
        device: Device for model.

    Returns:
        Dict scale -> {correlation, p_value, mean_hyperbolicity}.
    """
    if device is None:
        device = next(model.parameters()).device
    _, _, _, pred = evaluate_model(model, data, device)
    pred = pred.cpu().numpy()
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    correct = (pred == y) & test_mask
    node_correct = {i: bool(correct[i]) for i in np.where(test_mask)[0]}

    results: Dict[int, Dict[str, float]] = {}
    for radius in scales:
        hyp = compute_local_hyperbolicity_all(G, radius=radius)
        curv_list = []
        correct_list = []
        for nid in node_correct:
            if nid in hyp:
                curv_list.append(hyp[nid])
                correct_list.append(float(node_correct[nid]))
        if len(curv_list) < 5:
            results[radius] = {"correlation": 0.0, "p_value": 1.0, "mean_hyperbolicity": 0.0}
            continue
        r, p = stats.pearsonr(curv_list, correct_list)
        results[radius] = {
            "correlation": float(r),
            "p_value": float(p),
            "mean_hyperbolicity": float(np.mean(curv_list)),
        }
        logger.info("Scale (radius) %d: correlation=%.4f, p=%.4e, mean_hyperbolicity=%.4f",
                    radius, r, p, results[radius]["mean_hyperbolicity"])
    return results


def compare_rewiring_strategies(
    dataset_name: str,
    data: Data,
    G_curved: Any,
    edge_curvatures: Dict[tuple, float],
    model_class: Type[torch.nn.Module],
    num_runs: int = 5,
    epochs: Optional[int] = None,
    device: Optional[torch.device] = None,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """
    Run the full pipeline num_runs times with different seeds for: original graph,
    curvature-rewired graph, random-rewired graph. Report mean and std accuracy for each.

    Args:
        dataset_name: Name of dataset (for logging).
        data: PyG Data (original).
        G_curved: NetworkX graph with curvature (e.g. LCC from compute_ollivier_ricci).
        edge_curvatures: Dict (u,v) -> curvature.
        model_class: GCN or GAT class (not instance).
        num_runs: Number of runs per strategy.
        epochs: Training epochs; uses CONFIG["epochs"] if None.
        device: Device; if None, uses cpu/cuda.
        **model_kwargs: Passed to model_class (e.g. num_features, num_classes).

    Returns:
        DataFrame with columns: strategy, mean_test_acc, std_test_acc, (and optionally run accs).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = CONFIG["epochs"]
    lr = CONFIG["learning_rate"]
    threshold = CONFIG["rewiring_threshold"]
    num_edges_add = CONFIG["num_edges_add"]

    strategies = ["original", "curvature_rewired", "random_rewired"]
    results: List[Dict[str, Any]] = []

    for strategy in strategies:
        test_accs: List[float] = []
        for run in range(num_runs):
            seed = 42 + run
            torch.manual_seed(seed)
            np.random.seed(seed)
            if strategy == "original":
                d = data
            elif strategy == "curvature_rewired":
                d = curvature_based_rewiring(G_curved, edge_curvatures, data, threshold=threshold, num_edges_add=num_edges_add)
            else:
                d = random_rewiring(data, num_edges_add=num_edges_add, seed=seed)
            d = d.to(device)
            model = model_class(**model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_model(model, d, optimizer, epochs=epochs, device=device)
            _, _, test_acc, _ = evaluate_model(model, d, device)
            test_accs.append(test_acc)
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs))
        results.append({
            "strategy": strategy,
            "mean_test_acc": mean_acc,
            "std_test_acc": std_acc,
            "test_accs": test_accs,
        })
        logger.info("%s (%s): mean test acc = %.4f ± %.4f", dataset_name, strategy, mean_acc, std_acc)

    return pd.DataFrame(results)
