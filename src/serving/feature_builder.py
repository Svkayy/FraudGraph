import numpy as np

from ..features import normalize_card1, merchant_key, device_key, hour_of_day


class FeatureBuilder:
    """Reconstructs the 437-dim transaction node feature vector from a live
    payload, matching Stage 2's construction exactly:
      - numerics: median-impute unknown fields, then StandardScaler
      - categoricals: label code normalized to (code+1)/(cardinality+1), missing -> 0

    A live payload only carries the core fields; the Vesta V/C/D columns are
    unavailable at request time and fall back to the training median — a
    cold-start the graph neighbor embeddings compensate for.
    """

    def __init__(self, graph_meta: dict):
        self.num_cols = graph_meta["num_cols"]
        self.cat_cols = graph_meta["cat_cols"]
        self.cat_maps = graph_meta["cat_maps"]
        self.medians = graph_meta["medians"]
        self.scaler_mean = np.asarray(graph_meta["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(graph_meta["scaler_scale"], dtype=np.float64)

        self.num_index = {c: i for i, c in enumerate(self.num_cols)}
        self.median_vec = np.array([self.medians[c] for c in self.num_cols], dtype=np.float64)

    def build(self, payload: dict, velocity: dict) -> np.ndarray:
        known_num = {
            "TransactionAmt": float(payload["TransactionAmt"]),
            "TransactionDT": float(payload["TransactionDT"]),
            "card1": normalize_card1(payload.get("card1")),
            "addr1": float(payload["addr1"]) if payload.get("addr1") is not None else self.medians.get("addr1", -999.0),
            "hour": hour_of_day(float(payload["TransactionDT"])),
            "velocity_1h": float(velocity.get("count_1h", 1)),
            "velocity_6h": float(velocity.get("count_6h", 1)),
            "velocity_24h": float(velocity.get("count_24h", 1)),
        }

        num_vec = self.median_vec.copy()
        for col, val in known_num.items():
            if col in self.num_index:
                num_vec[self.num_index[col]] = val
        num_scaled = (num_vec - self.scaler_mean) / self.scaler_scale

        known_cat = {"ProductCD": payload.get("ProductCD"),
                     "DeviceInfo": device_key(payload.get("DeviceInfo"))}
        cat_vec = np.zeros(len(self.cat_cols), dtype=np.float64)
        for i, col in enumerate(self.cat_cols):
            raw = known_cat.get(col)
            if raw is None:
                continue
            m = self.cat_maps.get(col, {})
            if raw in m:
                cat_vec[i] = (m[raw] + 1.0) / (len(m) + 1.0)

        return np.hstack([num_scaled, cat_vec]).astype(np.float32)
