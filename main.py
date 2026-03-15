#!/usr/bin/env python3
"""
Single entry point for curvature-aware GNN experiments.
Runs full pipeline: load data, train baselines, compute curvature, rewiring, comparison, and save results.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import (
    compare_rewiring_strategies,
    curvature_accuracy_correlation,
    multiscale_analysis,
    per_node_analysis,
)
from src.curvature import (
    compute_local_hyperbolicity_all,
    compute_ollivier_ricci,
    get_node_curvature,
)
from src.models import GAT, GCN, evaluate_model, train_model
from src.rewiring import curvature_based_rewiring, random_rewiring
from src.utils import CONFIG, get_graph_stats, load_dataset, pyg_to_networkx
from src.visualization import (
    plot_curvature_distribution,
    plot_curvature_vs_accuracy,
    plot_graph_curvature,
    plot_multiscale_hyperbolicity,
    plot_rewiring_comparison,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Override config for --quick
    epochs = 50 if args.quick else args.epochs
    num_runs = 1 if args.quick else args.num_runs
    config = CONFIG.copy()
    config["epochs"] = epochs

    # 1) Load dataset and print graph stats
    data = load_dataset(args.dataset, root=os.path.join(os.path.dirname(__file__), "data"))
    data = data.to(device)
    G = pyg_to_networkx(data)
    get_graph_stats(G)

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    # 2) Train baseline GCN and GAT, store histories
    set_seeds(42)
    gcn = GCN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_gnn_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=config["learning_rate"])
    hist_gcn = train_model(gcn, data, optimizer_gcn, epochs=epochs, device=device)

    set_seeds(43)
    gat = GAT(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=2,
        dropout=config["gat_dropout"],
        heads=config["gat_heads"],
        head_dim=config["gat_hidden_per_head"],
    ).to(device)
    optimizer_gat = torch.optim.Adam(gat.parameters(), lr=config["learning_rate"])
    hist_gat = train_model(gat, data, optimizer_gat, epochs=epochs, device=device)

    # 3) Compute Ollivier-Ricci curvature
    G_curved, edge_curvatures = compute_ollivier_ricci(G, alpha=config["curvature_alpha"])

    # 4) Plot curvature distribution
    plot_curvature_distribution(edge_curvatures, args.dataset)

    # 5) Per-node curvature (for nodes in G_curved; map to full graph by using only LCC nodes)
    node_curvature = get_node_curvature(G_curved)
    # Extend to full graph: nodes not in G_curved get 0.0 or NaN; we use 0.0 for plotting
    for i in range(data.num_nodes):
        if i not in node_curvature:
            node_curvature[i] = 0.0

    # 6) Per-node analysis (curvature vs accuracy)
    df_per_node = per_node_analysis(gcn, data, node_curvature, device=device)
    curvature_accuracy_correlation(df_per_node)

    # 7) Plot curvature vs accuracy
    plot_curvature_vs_accuracy(df_per_node, args.dataset)

    # 8) Visualize graph with curvature coloring
    plot_graph_curvature(G_curved, node_curvature, args.dataset, sample_size=300)

    # 9) Local hyperbolicity (with progress bar)
    local_hyp = compute_local_hyperbolicity_all(G, radius=2)

    # 10) Multiscale analysis
    multiscale_results = multiscale_analysis(G, data, gcn, scales=[1, 2, 3], device=device)
    plot_multiscale_hyperbolicity(multiscale_results, args.dataset)

    # 11) Curvature-based rewiring
    data_curv_rewired = curvature_based_rewiring(
        G_curved,
        edge_curvatures,
        data,
        threshold=config["rewiring_threshold"],
        num_edges_add=config["num_edges_add"],
    )
    data_curv_rewired = data_curv_rewired.to(device)

    # 12) Train GCN on curvature-rewired graph
    set_seeds(44)
    gcn_curv = GCN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_gnn_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer_curv = torch.optim.Adam(gcn_curv.parameters(), lr=config["learning_rate"])
    train_model(gcn_curv, data_curv_rewired, optimizer_curv, epochs=epochs, device=device)

    # 13) Random rewiring baseline
    data_rand_rewired = random_rewiring(data, num_edges_add=config["num_edges_add"], seed=45)
    data_rand_rewired = data_rand_rewired.to(device)

    # 14) Train GCN on random-rewired graph
    set_seeds(46)
    gcn_rand = GCN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_gnn_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer_rand = torch.optim.Adam(gcn_rand.parameters(), lr=config["learning_rate"])
    train_model(gcn_rand, data_rand_rewired, optimizer_rand, epochs=epochs, device=device)

    # 15) Compare all: run compare_rewiring_strategies for mean±std over num_runs
    results_df = compare_rewiring_strategies(
        args.dataset,
        data,
        G_curved,
        edge_curvatures,
        GCN,
        num_runs=num_runs,
        epochs=epochs,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_gnn_layers"],
        dropout=config["dropout"],
    )

    # Summary table
    logger.info("=== Summary ===")
    print(results_df[["strategy", "mean_test_acc", "std_test_acc"]].to_string(index=False))

    # Training curves (GCN vs GAT baseline)
    plot_training_curves(
        [hist_gcn, hist_gat],
        ["GCN", "GAT"],
    )
    plot_rewiring_comparison(results_df, args.dataset)

    # 16) Save results to CSV in results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, f"rewiring_comparison_{args.dataset}.csv"), index=False)
    df_per_node.to_csv(os.path.join(results_dir, f"per_node_analysis_{args.dataset}.csv"), index=False)
    multiscale_df = pd.DataFrame([
        {"scale": k, **v} for k, v in multiscale_results.items()
    ])
    multiscale_df.to_csv(os.path.join(results_dir, f"multiscale_hyperbolicity_{args.dataset}.csv"), index=False)
    logger.info("Results saved to %s", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvature-aware GNN analysis pipeline")
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"],
                        help="Dataset name (default: Cora)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs for rewiring comparison (default: 3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--quick", action="store_true", help="Quick run: 50 epochs, 1 run")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
