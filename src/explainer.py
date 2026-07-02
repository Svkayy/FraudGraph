from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Reason:
    factor: str
    weight: float
    edge_type: str
    evidence: dict[str, Any]


class FraudExplainer:
    """Deterministic, rule-based explanations grounded in graph structure and
    node-feature outliers. Reasons are ranked by an evidence-driven weight so
    the top few are the most informative. Complements — not replaces — the
    Hybrid model's attention weights, and is fast enough to serve inline.

    Design notes learned from the first-pass eval:
      - The UNKNOWN device node aggregates ~80% of transactions (identity file
        coverage is only ~20%). It's noise, not signal — we skip it entirely.
      - A card with many fraud transactions at BASELINE rate is a volume
        artifact, not a risk signal. Reasons require actual lift over baseline.
      - Below-baseline lift is a protective signal, phrased distinctly from
        risky ones so a fraud analyst can read them correctly.
      - Weights scale with lift (relative signal), not fraud_count (volume)."""

    RISKY_LIFT_MIN = 1.5
    PROTECTIVE_LIFT_MAX = 0.5
    MIN_HISTORY = 3

    def __init__(
        self,
        card_stats: dict[int, dict[str, float]],
        merchant_stats: dict[int, dict[str, float]],
        device_stats: dict[int, dict[str, float]],
        amt_mean: float,
        amt_std: float,
        train_fraud_rate: float,
        unknown_device_idx: int | None = None,
    ):
        self.card_stats = card_stats
        self.merchant_stats = merchant_stats
        self.device_stats = device_stats
        self.amt_mean = amt_mean
        self.amt_std = amt_std
        self.train_fraud_rate = train_fraud_rate
        self.unknown_device_idx = unknown_device_idx

    @staticmethod
    def _lift_weight(lift: float, base: float = 0.1, scale: float = 0.35, cap: float = 1.0) -> float:
        return min(cap, base + scale * max(0.0, lift - 1.0))

    def explain(
        self,
        transaction_id: str | int,
        fraud_score: float,
        card_idx: int | None,
        merchant_idx: int | None,
        device_idx: int | None,
        amount: float,
        velocity_neighbors: list[dict] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        reasons: list[Reason] = []

        # ---------- Card ----------
        card = self.card_stats.get(card_idx) if card_idx is not None else None
        if card and card["tx_count"] >= self.MIN_HISTORY:
            lift = card["fraud_rate"] / max(self.train_fraud_rate, 1e-6)
            if lift >= self.RISKY_LIFT_MIN and card["fraud_count"] >= 2:
                reasons.append(Reason(
                    factor=(f"shared card with elevated fraud rate: "
                            f"{card['fraud_rate']*100:.1f}% "
                            f"({lift:.1f}x baseline, {int(card['fraud_count'])} prior fraud tx)"),
                    weight=self._lift_weight(lift),
                    edge_type="uses_card",
                    evidence={"card_idx": card_idx, **card, "lift": round(lift, 2)},
                ))
            elif lift <= self.PROTECTIVE_LIFT_MAX and card["tx_count"] >= 10:
                reasons.append(Reason(
                    factor=(f"protective: card has below-baseline fraud rate "
                            f"({card['fraud_rate']*100:.1f}%, {lift:.2f}x baseline, "
                            f"{int(card['tx_count'])} prior tx)"),
                    weight=0.15,
                    edge_type="uses_card",
                    evidence={"card_idx": card_idx, **card, "lift": round(lift, 2), "protective": True},
                ))

        # ---------- Device ----------
        # Skip the UNKNOWN placeholder — it's the identity-missing bucket and
        # applies to ~80% of transactions with no signal.
        if device_idx is not None and device_idx != self.unknown_device_idx:
            dev = self.device_stats.get(device_idx)
            if dev and dev["tx_count"] >= self.MIN_HISTORY:
                lift = dev["fraud_rate"] / max(self.train_fraud_rate, 1e-6)
                if lift >= self.RISKY_LIFT_MIN and dev["fraud_count"] >= 2:
                    reasons.append(Reason(
                        factor=(f"device linked to {int(dev['unique_cards'])} cards, "
                                f"{int(dev['fraud_count'])} with prior fraud "
                                f"({lift:.1f}x baseline)"),
                        weight=self._lift_weight(lift),
                        edge_type="on_device",
                        evidence={"device_idx": device_idx, **dev, "lift": round(lift, 2)},
                    ))

        # ---------- Merchant ----------
        merch = self.merchant_stats.get(merchant_idx) if merchant_idx is not None else None
        if merch and merch["tx_count"] > 20:
            lift = merch["fraud_rate"] / max(self.train_fraud_rate, 1e-6)
            if lift >= self.RISKY_LIFT_MIN:
                reasons.append(Reason(
                    factor=(f"merchant has elevated fraud rate: "
                            f"{merch['fraud_rate']*100:.1f}% ({lift:.1f}x baseline)"),
                    weight=self._lift_weight(lift, scale=0.25, cap=0.7),
                    edge_type="at_merchant",
                    evidence={"merchant_idx": merchant_idx, **merch, "lift": round(lift, 2)},
                ))

        # ---------- Velocity cluster ----------
        if velocity_neighbors and len(velocity_neighbors) >= 2:
            n_fraud = sum(1 for n in velocity_neighbors if n.get("is_fraud"))
            if n_fraud > 0:
                reasons.append(Reason(
                    factor=(f"card-testing burst: {len(velocity_neighbors)} tx on this card in the "
                            f"last 5 minutes, {n_fraud} already flagged fraud"),
                    weight=min(1.0, 0.5 + 0.15 * n_fraud),
                    edge_type="velocity_cluster",
                    evidence={"cluster_size": len(velocity_neighbors), "fraud_in_cluster": n_fraud},
                ))
            elif len(velocity_neighbors) >= 4:
                reasons.append(Reason(
                    factor=(f"card-testing burst: {len(velocity_neighbors)} tx on this card in the "
                            f"last 5 minutes"),
                    weight=0.3,
                    edge_type="velocity_cluster",
                    evidence={"cluster_size": len(velocity_neighbors), "fraud_in_cluster": 0},
                ))

        # ---------- Amount outlier ----------
        if self.amt_std > 0:
            z = (amount - self.amt_mean) / self.amt_std
            if abs(z) > 3.0:
                reasons.append(Reason(
                    factor=f"amount {abs(z):.1f}σ {'above' if z > 0 else 'below'} training-set mean",
                    weight=min(0.5, 0.05 + 0.05 * (abs(z) - 3)),
                    edge_type="self",
                    evidence={"amount": amount, "z_score": round(float(z), 2)},
                ))

        # ---------- Fallback ----------
        if not reasons:
            if fraud_score < 0.3:
                reasons.append(Reason(
                    factor="no elevated-risk signals in card, merchant, or device history",
                    weight=1.0, edge_type="none", evidence={},
                ))
            else:
                reasons.append(Reason(
                    factor="model flagged based on feature combination without a single dominant relational cause",
                    weight=1.0, edge_type="none", evidence={},
                ))

        reasons.sort(key=lambda r: r.weight, reverse=True)
        total = sum(r.weight for r in reasons) or 1.0
        for r in reasons:
            r.weight = round(r.weight / total, 3)

        return {
            "transaction_id": transaction_id,
            "fraud_score": round(float(fraud_score), 4),
            "top_reasons": [asdict(r) for r in reasons[:top_k]],
        }
