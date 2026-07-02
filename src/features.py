"""Canonical feature and entity-key semantics shared by the Kaggle notebooks
and the serving pipeline. Any drift between training-time and serving-time
definitions silently corrupts features, so both sides import from here.

Semantics that MUST stay consistent between training and serving:

1. Velocity counts INCLUDE the current transaction. Notebook 01 computes them
   with a rolling window over the time-sorted frame, where the window contains
   the row itself (count_1h >= 1 always). Serving must therefore
   increment-then-read the Redis counter, not read-then-increment.

2. card1 keys are FLOATS. The raw column is float64 and missing values map to
   -999.0, so the card map is keyed by float (e.g. 7919.0, -999.0). Coerce any
   incoming payload value with `normalize_card1` before lookup.

3. Merchant identity requires addr1. Transactions with missing addr1 get NO
   merchant edge/stats (mirroring how missing DeviceInfo gets no device edge).
   Lumping them into a per-product "-999" pseudo-merchant creates mega-hub
   nodes that carry no signal.

4. Device identity for missing DeviceInfo maps to the single UNKNOWN key. The
   UNKNOWN device exists as a graph node but is excluded from explanations.
"""

from __future__ import annotations

DEVICE_UNKNOWN = "UNKNOWN"
CARD_MISSING = -999.0


def normalize_card1(value) -> float:
    """Coerce a payload card1 (int, float, str, or None) to the canonical
    float key used in card_map."""
    if value is None:
        return CARD_MISSING
    try:
        return float(value)
    except (TypeError, ValueError):
        return CARD_MISSING


def merchant_key(product_cd, addr1) -> str | None:
    """Canonical merchant key, or None when addr1 is missing (no merchant
    identity without an address region)."""
    if product_cd is None or addr1 is None:
        return None
    try:
        addr = float(addr1)
    except (TypeError, ValueError):
        return None
    if addr == CARD_MISSING:
        return None
    return f"{product_cd}|{addr}"


def device_key(device_info) -> str:
    """Canonical device key; missing/empty DeviceInfo maps to UNKNOWN."""
    if device_info is None or (isinstance(device_info, str) and not device_info.strip()):
        return DEVICE_UNKNOWN
    return str(device_info)


def hour_of_day(transaction_dt: float) -> int:
    """Hour of day from TransactionDT (seconds from dataset reference)."""
    return int((transaction_dt // 3600) % 24)
