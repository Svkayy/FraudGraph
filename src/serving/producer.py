import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

from . import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test_sample_full.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rate", type=float, default=20.0, help="messages per second")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.n)
    fcols = sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    delay = 1.0 / args.rate
    for _, r in df.iterrows():
        payload = {
            "TransactionAmt": float(r["TransactionAmt"]),
            "ProductCD": str(r["ProductCD"]),
            "card1": float(r["card1"]),
            "addr1": None if pd.isna(r["addr1"]) else float(r["addr1"]),
            "DeviceInfo": None if pd.isna(r.get("DeviceInfo")) else str(r["DeviceInfo"]),
            "TransactionDT": float(r["TransactionDT"]),
            "TransactionID": int(r["TransactionID"]),
        }
        if fcols:
            payload["feature_vector"] = [float(r[c]) for c in fcols]
        producer.send(config.TOPIC_IN, payload)
        time.sleep(delay)
    producer.flush()
    print(f"sent {len(df)} transactions to '{config.TOPIC_IN}'")


if __name__ == "__main__":
    main()
