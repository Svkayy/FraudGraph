import pickle
import time

import numpy as np
import torch

from . import config
from .feature_builder import FeatureBuilder
from ..features import normalize_card1, merchant_key, device_key
from ..model import HybridGatedGNN, PureSAGE
from ..explainer import FraudExplainer

_ARCH = {"HybridGatedGNN": HybridGatedGNN, "PureSAGE": PureSAGE}


class FraudScorer:
    """Loads the trained GNN and scores a single transaction by assembling a
    small local subgraph (the transaction + its card / merchant / device
    neighbors) and running the real message-passing forward. Neighbor nodes are
    initialized from the model's learned entity-embedding tables — the 2-hop
    neighborhood context each entity accumulated during training is summarized
    in those vectors, which is what makes single-transaction inference fast
    enough to serve inline."""

    def __init__(self):
        with open(config.MODEL_META, "rb") as f:
            self.meta = pickle.load(f)
        with open(config.GRAPH_META, "rb") as f:
            self.graph_meta = pickle.load(f)
        with open(config.EXPLAINER_REF, "rb") as f:
            ref = pickle.load(f)

        arch = _ARCH.get(self.meta.get("winner_arch", "HybridGatedGNN"), HybridGatedGNN)
        self.model = arch(
            in_dim=self.meta["in_dim"],
            num_cards=self.meta["num_cards"],
            num_merchants=self.meta["num_merchants"],
            num_devices=self.meta["num_devices"],
            metadata=self.meta["metadata"],
            hidden_dim=self.meta["hidden_dim"],
            out_dim=self.meta["out_dim"],
            dropout=self.meta["dropout"],
        )
        state = torch.load(config.MODEL_WEIGHTS, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        self.card_mean = self.model.card_emb.weight.detach().mean(0, keepdim=True)
        self.merch_mean = self.model.merchant_emb.weight.detach().mean(0, keepdim=True)
        self.dev_mean = self.model.device_emb.weight.detach().mean(0, keepdim=True)

        self.fb = FeatureBuilder(self.graph_meta)
        self.card_map = ref["card_map"]
        self.merch_map = ref["merch_map"]
        self.dev_map = ref["dev_map"]
        self.card_history = ref["card_history"]
        self.velocity_window = ref["velocity_window_sec"]
        self.explainer = FraudExplainer(
            card_stats=ref["card_stats"], merchant_stats=ref["merchant_stats"],
            device_stats=ref["device_stats"], amt_mean=ref["amt_mean"],
            amt_std=ref["amt_std"], train_fraud_rate=ref["train_fraud_rate"],
            unknown_device_idx=ref.get("unknown_device_idx"),
        )

    def _entity_indices(self, payload):
        card_idx = self.card_map.get(normalize_card1(payload.get("card1")))
        mkey = merchant_key(payload.get("ProductCD"), payload.get("addr1"))
        merch_idx = self.merch_map.get(mkey) if mkey is not None else None
        dev_idx = self.dev_map.get(device_key(payload.get("DeviceInfo")))
        return card_idx, merch_idx, dev_idx

    def _emb(self, table, mean, idx):
        if idx is None:
            return mean
        return table.weight.detach()[idx:idx + 1]

    @torch.no_grad()
    def _gnn_score(self, x_np, card_idx, merch_idx, dev_idx):
        x = torch.from_numpy(x_np).unsqueeze(0)
        h = {
            "transaction": self.model.tx_encoder(x),
            "card": self._emb(self.model.card_emb, self.card_mean, card_idx),
            "device": self._emb(self.model.device_emb, self.dev_mean, dev_idx),
        }
        e = torch.tensor([[0], [0]], dtype=torch.long)
        edges = {
            ("transaction", "uses_card", "card"): e,
            ("card", "rev_uses_card", "transaction"): e,
            ("transaction", "on_device", "device"): e,
            ("device", "rev_on_device", "transaction"): e,
        }
        if merch_idx is not None:
            h["merchant"] = self._emb(self.model.merchant_emb, self.merch_mean, merch_idx)
            edges[("transaction", "at_merchant", "merchant")] = e
            edges[("merchant", "rev_at_merchant", "transaction")] = e

        h = self.model.conv1(h, edges)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.model.conv2(h, edges)
        h = {k: torch.relu(v) for k, v in h.items()}
        logit = self.model.head(h["transaction"]).squeeze(-1)
        return float(torch.sigmoid(logit).item())

    def _velocity_neighbors(self, card_idx, ts):
        if card_idx is None:
            return []
        hist = self.card_history.get(card_idx) or self.card_history.get(str(card_idx)) or []
        lo = ts - self.velocity_window
        return [{"is_fraud": bool(f)} for (t, f) in hist if lo <= t < ts]

    def score(self, payload: dict, velocity: dict, feature_vector=None) -> dict:
        t0 = time.perf_counter()
        card_idx, merch_idx, dev_idx = self._entity_indices(payload)
        if feature_vector is not None:
            # feature store supplied the materialized 437-vector
            x = np.asarray(feature_vector, dtype=np.float32)
        else:
            # minimal payload: known fields + median-imputed remainder
            x = self.fb.build(payload, velocity)
        fraud_score = self._gnn_score(x, card_idx, merch_idx, dev_idx)

        explanation = self.explainer.explain(
            transaction_id=payload.get("TransactionID", "unknown"),
            fraud_score=fraud_score,
            card_idx=card_idx, merchant_idx=merch_idx, device_idx=dev_idx,
            amount=float(payload["TransactionAmt"]),
            velocity_neighbors=self._velocity_neighbors(card_idx, float(payload["TransactionDT"])),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "transaction_id": str(payload.get("TransactionID", "unknown")),
            "fraud_score": round(fraud_score, 4),
            "decision": config.decision_for(fraud_score),
            "latency_ms": round(latency_ms, 2),
            "top_reasons": explanation["top_reasons"],
        }
