import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.getenv("FRAUDGRAPH_ARTIFACTS", REPO_ROOT / "artifacts"))

MODEL_WEIGHTS = ARTIFACTS / "graphsage_hybrid.pt"
MODEL_META = ARTIFACTS / "model_meta.pkl"
GRAPH_META = ARTIFACTS / "graph_meta.pkl"
EXPLAINER_REF = ARTIFACTS / "explainer_reference.pkl"

# Decision thresholds (fixed for the project; production would tune to cost).
APPROVE_BELOW = 0.30
DECLINE_ABOVE = 0.70


def decision_for(score: float) -> str:
    if score < APPROVE_BELOW:
        return "APPROVE"
    if score > DECLINE_ABOVE:
        return "DECLINE"
    return "FLAG"


# Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_IN = "transactions"
TOPIC_OUT = "fraud-scores"
NUM_PARTITIONS = 3
CONSUMER_GROUP = "fraud-scorer"

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VELOCITY_TTL_SEC = 48 * 3600
NEIGHBOR_LIST_LEN = 20

# Velocity windows (seconds)
WINDOWS = {"count_1h": 3600, "count_6h": 21600, "count_24h": 86400}
