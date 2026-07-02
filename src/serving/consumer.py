import json

from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import start_http_server

from . import config, metrics
from .api import score_transaction, state
from .inference import FraudScorer
from .redis_store import RedisStore
from .kafka_admin import ensure_topics


def run():
    if not ensure_topics():
        raise SystemExit(f"No Kafka broker at {config.KAFKA_BOOTSTRAP}. Is docker compose up?")

    # standalone process: populate the shared state the scoring path reads
    scorer = state.setdefault("scorer", FraudScorer())
    state.setdefault("redis", RedisStore())
    metrics.configure(scorer.explainer.amt_mean, scorer.explainer.amt_std)
    start_http_server(8001)  # Prometheus scrapes the consumer here
    print("consumer metrics on :8001/metrics")

    consumer = KafkaConsumer(
        config.TOPIC_IN,
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        group_id=config.CONSUMER_GROUP,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"consumer up on '{config.TOPIC_IN}' -> '{config.TOPIC_OUT}'")
    for msg in consumer:
        result = score_transaction(msg.value)
        producer.send(config.TOPIC_OUT, result)
        print(f"  {result['transaction_id']}  {result['decision']:8s}  "
              f"score={result['fraud_score']:.3f}  {result['latency_ms']:.1f}ms")


if __name__ == "__main__":
    run()
