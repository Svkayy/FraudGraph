import argparse
import time

import numpy as np
import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test_sample_full.csv")
    ap.add_argument("--url", default="http://localhost:8000/score")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fcols = sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    rows = df.sample(min(args.n, len(df)), replace=len(df) < args.n, random_state=0)

    latencies, decisions = [], {"APPROVE": 0, "FLAG": 0, "DECLINE": 0}
    errors = 0
    print(f"sending {len(rows)} transactions to {args.url} "
          f"({'with feature vectors' if fcols else 'core fields only'}) ...")
    t_start = time.perf_counter()

    for _, r in rows.iterrows():
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
        t0 = time.perf_counter()
        try:
            resp = requests.post(args.url, json=payload, timeout=5)
            resp.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            decisions[resp.json()["decision"]] += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print("  error:", e)

    wall = time.perf_counter() - t_start
    lat = np.array(latencies)
    print(f"\n{'='*46}\nRESULTS ({len(lat)} ok, {errors} errors)\n{'-'*46}")
    if len(lat):
        print(f"throughput:  {len(lat)/wall:.1f} req/s")
        print(f"latency p50: {np.percentile(lat,50):.1f} ms")
        print(f"latency p95: {np.percentile(lat,95):.1f} ms")
        print(f"latency p99: {np.percentile(lat,99):.1f} ms")
        print(f"latency max: {lat.max():.1f} ms")
        print(f"decisions:   {decisions}")


if __name__ == "__main__":
    main()
