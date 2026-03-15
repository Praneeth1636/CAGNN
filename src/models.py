"""
GCN and GAT implementations and training/evaluation utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from .utils import CONFIG

logger = logging.getLogger(__name__)


class GCN(nn.Module):
    """
    Graph Convolutional Network with configurable number of layers.
    Uses GCNConv, ReLU and dropout between layers, log_softmax at the output.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Graph Attention Network: first layer 8 heads x 8 hidden dims, second layer 1 head to num_classes.
    Uses ELU activation and dropout=0.6.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.6,
        heads: int = 8,
        head_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        # First layer: 8 heads, 8 hidden dims each -> 8*8 = 64 output
        self.convs.append(
            GATConv(num_features, head_dim, heads=heads, dropout=dropout)
        )
        # Second layer: 1 head, output num_classes
        self.convs.append(
            GATConv(heads * head_dim, num_classes, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.elu(conv(x, edge_index))
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


def train_model(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    epochs: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Train a GNN model with cross-entropy loss on the training mask.
    Logs every 50 epochs.

    Args:
        model: GCN or GAT model.
        data: PyG Data with x, edge_index, y, train_mask, val_mask, test_mask.
        optimizer: Torch optimizer (e.g. Adam).
        epochs: Number of training epochs.
        device: Device to run on; if None, uses data's device.

    Returns:
        Dict with keys: train_loss, val_acc, test_acc (lists of per-epoch values).
    """
    if device is None:
        device = next(model.parameters()).device
    model.train()
    history: Dict[str, List[float]] = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        history["train_loss"].append(loss.item())
        with torch.no_grad():
            model.eval()
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)
            model.train()

        if (epoch + 1) % 50 == 0:
            logger.info(
                "Epoch %04d | Loss: %.4f | Val: %.4f | Test: %.4f",
                epoch + 1, loss.item(), val_acc, test_acc,
            )

    return history


def evaluate_model(
    model: nn.Module,
    data: Data,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float, torch.Tensor]:
    """
    Evaluate model on train/val/test splits and return per-node predictions.

    Args:
        model: Trained GNN model.
        data: PyG Data with masks and y.
        device: Device; if None, inferred from model.

    Returns:
        (train_acc, val_acc, test_acc, predictions) where predictions is a 1D LongTensor of shape [num_nodes].
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return train_acc, val_acc, test_acc, pred
