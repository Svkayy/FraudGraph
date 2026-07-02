import time
from collections import deque

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

SCORE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LATENCY_BUCKETS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

fraud_score = Histogram(
    "fraud_score", "Distribution of fraud scores", buckets=SCORE_BUCKETS)
decision_total = Counter(
    "decision_total", "Decisions by type", ["decision"])
inference_latency_seconds = Histogram(
    "inference_latency_seconds", "End-to-end scoring latency (seconds)", buckets=LATENCY_BUCKETS)
transaction_amount_mean = Gauge(
    "transaction_amount_mean", "5-minute rolling mean of TransactionAmt")
train_amount_mean = Gauge(
    "train_amount_mean", "Training-set mean TransactionAmt (drift reference)")
feature_drift_alert = Gauge(
    "feature_drift_alert", "1 when rolling amount mean drifts >2 sigma from training mean")

_WINDOW_SEC = 300
_window = deque()
_train = {"mean": None, "std": None}


def configure(train_mean: float, train_std: float) -> None:
    _train["mean"] = float(train_mean)
    _train["std"] = float(train_std)
    train_amount_mean.set(float(train_mean))


def observe(score: float, decision: str, latency_ms: float, amount: float, now: float | None = None) -> None:
    now = time.time() if now is None else now
    fraud_score.observe(score)
    decision_total.labels(decision=decision).inc()
    inference_latency_seconds.observe(latency_ms / 1000.0)

    _window.append((now, float(amount)))
    cutoff = now - _WINDOW_SEC
    while _window and _window[0][0] < cutoff:
        _window.popleft()

    mean_amt = sum(a for _, a in _window) / len(_window)
    transaction_amount_mean.set(mean_amt)
    if _train["std"]:
        drifted = abs(mean_amt - _train["mean"]) > 2 * _train["std"]
        feature_drift_alert.set(1 if drifted else 0)


def render():
    return generate_latest(), CONTENT_TYPE_LATEST
