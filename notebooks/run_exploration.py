#!/usr/bin/env python3
"""Run all code cells from exploration.ipynb."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils import load_dataset, pyg_to_networkx, get_graph_stats, CONFIG
import logging
logging.basicConfig(level=logging.INFO)

data = load_dataset("Cora", root=os.path.join(ROOT, "data"))
G = pyg_to_networkx(data)
get_graph_stats(G)
print("Nodes:", data.num_nodes, "Edges:", data.edge_index.size(1), "Features:", data.num_node_features, "Classes:", data.y.max().item() + 1)

import networkx as nx
import numpy as np

nodes_0 = list(nx.ego_graph(G, 0, radius=2).nodes())
G_small = G.subgraph(nodes_0).copy()
print("Subgraph nodes:", G_small.number_of_nodes(), "edges:", G_small.number_of_edges())

from src.curvature import compute_ollivier_ricci, get_node_curvature

G_curved, edge_curvatures = compute_ollivier_ricci(G_small, alpha=0.5)
node_curv = get_node_curvature(G_curved)
print("Sample edge curvatures:", list(edge_curvatures.items())[:5])
print("Mean node curvature (sample):", np.mean(list(node_curv.values())))

from src.visualization import plot_graph_curvature

plot_graph_curvature(G_curved, node_curv, "Cora_subgraph", sample_size=500)
print("Saved to figures/graph_curvature_Cora_subgraph.png")

from src.rewiring import curvature_based_rewiring, random_rewiring
from src.models import GCN, train_model, evaluate_model
import torch

from src.curvature import compute_ollivier_ricci
G_full_curved, edge_curv_full = compute_ollivier_ricci(G, alpha=CONFIG["curvature_alpha"])

data_curv = curvature_based_rewiring(G_full_curved, edge_curv_full, data, threshold=CONFIG["rewiring_threshold"], num_edges_add=CONFIG["num_edges_add"])
data_rand = random_rewiring(data, num_edges_add=CONFIG["num_edges_add"], seed=42)

print("Original edges:", data.edge_index.size(1)//2, "Curvature-rewired:", data_curv.edge_index.size(1)//2, "Random-rewired:", data_rand.edge_index.size(1)//2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_g = data.to(device)
data_curv_g = data_curv.to(device)

num_features = data.num_node_features
num_classes = int(data.y.max().item()) + 1

for name, d in [("Original", data_g), ("Curvature-rewired", data_curv_g)]:
    model = GCN(num_features, num_classes, hidden_dim=CONFIG["hidden_dim"], num_layers=CONFIG["num_gnn_layers"], dropout=CONFIG["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    train_model(model, d, opt, epochs=50, device=device)
    train_acc, val_acc, test_acc, _ = evaluate_model(model, d, device)
    print(f"{name}: Test accuracy = {test_acc:.4f}")

print("Done.")
