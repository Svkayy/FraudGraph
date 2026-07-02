import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv


class HybridGatedConv(nn.Module):
    """Per-edge-type conv: run SAGE and GAT in parallel, mix via a learned
    sigmoid gate over the destination node's own features. A temperature
    parameter sharpens the gate as training progresses."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert out_dim % heads == 0 or heads == 1
        per_head = out_dim // heads if heads > 1 else out_dim
        self.sage = SAGEConv((in_dim, in_dim), out_dim, aggr="mean")
        self.gat = GATv2Conv(
            (in_dim, in_dim), per_head,
            heads=heads, concat=(heads > 1),
            add_self_loops=False, dropout=dropout,
        )
        self.gate = nn.Linear(in_dim, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))  # temp = exp(log_temp)

    def forward(self, x, edge_index):
        h_sage = self.sage(x, edge_index)
        h_gat = self.gat(x, edge_index)
        x_dst = x[1] if isinstance(x, tuple) else x
        g = torch.sigmoid(self.gate(x_dst) / self.log_temp.exp())
        return g * h_gat + (1.0 - g) * h_sage

    @torch.no_grad()
    def gate_stats(self, x, edge_index):
        x_dst = x[1] if isinstance(x, tuple) else x
        g = torch.sigmoid(self.gate(x_dst) / self.log_temp.exp())
        return g.mean().item(), g.std().item()


class _BaseFraudGNN(nn.Module):
    def __init__(self, in_dim, num_cards, num_merchants, num_devices,
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tx_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.card_emb     = nn.Embedding(num_cards, hidden_dim)
        self.merchant_emb = nn.Embedding(num_merchants, hidden_dim)
        self.device_emb   = nn.Embedding(num_devices, hidden_dim)

    def _init(self, x_tx, node_idx):
        h = {"transaction": self.tx_encoder(x_tx)}
        h["card"]     = self.card_emb.weight     if node_idx.get("card")     is None else self.card_emb(node_idx["card"])
        h["merchant"] = self.merchant_emb.weight if node_idx.get("merchant") is None else self.merchant_emb(node_idx["merchant"])
        h["device"]   = self.device_emb.weight   if node_idx.get("device")   is None else self.device_emb(node_idx["device"])
        return h


class PureSAGE(_BaseFraudGNN):
    """Baseline: SAGEConv on every edge type."""

    def __init__(self, in_dim, num_cards, num_merchants, num_devices, metadata,
                 hidden_dim=128, out_dim=64, dropout=0.2):
        super().__init__(in_dim, num_cards, num_merchants, num_devices, hidden_dim, dropout)
        self.out_dim = out_dim
        edge_types = metadata[1]
        self.conv1 = HeteroConv(
            {et: SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean") for et in edge_types},
            aggr="sum")
        self.conv2 = HeteroConv(
            {et: SAGEConv((hidden_dim, hidden_dim), out_dim, aggr="mean") for et in edge_types},
            aggr="sum")
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def encode(self, x_tx, edge_index_dict, node_idx=None):
        h = self._init(x_tx, node_idx or {})
        h = self.conv1(h, edge_index_dict); h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict); h = {k: F.relu(v) for k, v in h.items()}
        return h

    def forward(self, x_tx, edge_index_dict, node_idx=None):
        return self.head(self.encode(x_tx, edge_index_dict, node_idx)["transaction"]).squeeze(-1)


class HybridGatedGNN(_BaseFraudGNN):
    """Hybrid: on every edge type, run SAGE and GAT in parallel and mix them
    via a learned per-node gate. The model decides per-transaction whether
    attention or mean aggregation is more informative for that node's context."""

    def __init__(self, in_dim, num_cards, num_merchants, num_devices, metadata,
                 hidden_dim=128, out_dim=64, heads=4, dropout=0.2):
        super().__init__(in_dim, num_cards, num_merchants, num_devices, hidden_dim, dropout)
        self.out_dim = out_dim
        edge_types = metadata[1]
        self.conv1 = HeteroConv(
            {et: HybridGatedConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
             for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv(
            {et: HybridGatedConv(hidden_dim, out_dim, heads=1, dropout=dropout)
             for et in edge_types}, aggr="sum")
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def encode(self, x_tx, edge_index_dict, node_idx=None):
        h = self._init(x_tx, node_idx or {})
        h = self.conv1(h, edge_index_dict); h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict); h = {k: F.relu(v) for k, v in h.items()}
        return h

    def forward(self, x_tx, edge_index_dict, node_idx=None):
        return self.head(self.encode(x_tx, edge_index_dict, node_idx)["transaction"]).squeeze(-1)


def focal_bce_with_logits(logits, targets, alpha: float = 0.25, gamma: float = 2.0,
                          pos_weight: float | None = None):
    """Focal loss (Lin et al. 2017) with class-weighting on top.

    - `gamma` downweights easy examples so training focuses on the hard
      fraud/legit boundary rather than the many-easy-negatives majority.
    - `alpha` gives extra weight to the positive class.
    - Optional `pos_weight` composes with alpha for extreme imbalance.

    Better fit than plain weighted-BCE for a 3.5% positive class because it
    directly addresses the aggressive-gradient spikes that cause the noisy
    P@10R oscillations we observed with weighted-BCE."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=logits.device),
    )
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()
