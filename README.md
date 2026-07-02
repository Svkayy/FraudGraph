# FraudGraph — Real-Time Payment Fraud Detection with Graph Neural Networks

Fraud is **relational**: one stolen card hits many merchants, one device fingerprint
spans many accounts, one ring rotates through cards in minutes. A row-at-a-time tabular
model is blind to those links. FraudGraph models the IEEE-CIS transaction stream as a
**heterogeneous graph** (transactions, cards, merchants, devices) and scores each
transaction with a **hybrid GraphSAGE + attention** GNN — served in real time over a
Kafka + Redis pipeline at **~16 ms p99**, with a **human-readable reason** attached to
every decision.

---

## Architecture

```
  TRAINING (Kaggle GPU, offline)                 SERVING (local, real-time)
  ─────────────────────────────                  ──────────────────────────
  IEEE-CIS CSVs                                   Transaction event
     │  temporal 70/10/20 split                        │
     ▼                                                 ├── POST /score (sync) ──┐
  EDA + XGBoost baselines (MLflow)                     ▼                        │
     │                                          Kafka "transactions" → consumer │
     ▼                                                 │            │           │
  HeteroData graph  ──► GraphSAGE+GAT training         │      FraudScorer ◄── FastAPI
     │                        │                        │       ├─ Redis (velocity cache)
     ▼                        ▼                        │       ├─ GNN subgraph forward
  graph.pt          graphsage_hybrid.pt + embeddings   │       └─ Explainer (reasons)
                             │                          ▼
                    artifacts/ ───────────────► Kafka "fraud-scores"
                                                        │
                              Prometheus ◄── /metrics ──┘ ──► Grafana (localhost:3000)
```

---

## Results (temporal test set, leakage-free protocol)

| Model | AUC | Avg Precision | Precision @ 10% recall |
|---|---|---|---|
| XGBoost (lean, 8 features) | 0.769 | 0.139 | 0.250 |
| XGBoost (full, ~430 features) | **0.907** | **0.524** | 0.946 |
| GraphSAGE (pure) | 0.810 | 0.408 | 0.908 |
| **GraphSAGE + GAT (hybrid, served)** | 0.854 | 0.457 | **0.955** |
| XGBoost + GNN embeddings (stack) | 0.872 | 0.502 | 0.964 |

**How to read this honestly:**
- The hybrid GNN **matches the strongest tabular baseline on precision @ 10% recall**
  (0.955 vs 0.946) — the metric a fraud-ops team actually optimizes — and cleanly beats
  pure GraphSAGE on all three metrics (a controlled ablation: same graph, the learned gate
  is the only change).
- XGBoost-full keeps a **global AUC edge** because IEEE-CIS ships 339 heavily pre-engineered
  Vesta features that already encode much of the relational signal.
- The **GNN-embeddings-into-XGBoost stack did not help** under the strict protocol
  (train/serve embedding distribution mismatch) — reported as a negative result rather than
  buried.

**Serving latency** (100-transaction load test, CPU / Apple Silicon):
`p50 10.5 ms · p95 14.2 ms · p99 15.6 ms` — well under the 100 ms target.

---

## Key design decisions

1. **GNN over tabular ML.** Coordinated fraud lives in the *connections* between
   transactions (shared card/device/merchant, rapid bursts). A GNN aggregates signal from a
   transaction's neighbourhood; a tree model sees each row in isolation.

2. **Hybrid gated SAGE + GAT over pure GraphSAGE.** SAGE (mean aggregation) is the
   fraud-detection industry default — robust and scalable on dense edges. But the
   `velocity_cluster` edges are sparse and noisy (1.6× fraud lift). We run SAGE **and**
   GATv2 in parallel on every edge type and blend them with a **learned per-node gate**.
   The gate learned to apply attention exactly where it helps (velocity edges + card-history
   aggregation), and the ablation confirmed it beats pure SAGE by ~0.04 AUC.

3. **The `velocity_cluster` edge type (the differentiator).** A directed transaction→
   transaction edge connecting two payments on the same card within 5 minutes. It targets
   card-testing bursts that tabular models can't see relationally. Not from any tutorial.

Supporting choices: **strict temporal split** (never random — that leaks the future);
**focal loss** for the 3.5%-positive class; **precomputed entity embeddings** so a single
transaction scores in ~2 ms without re-sampling the whole graph.

---

## Explainability

Every decision carries the top graph-grounded reasons, e.g.:

```json
{ "fraud_score": 0.81, "decision": "DECLINE",
  "top_reasons": [
    { "factor": "card-testing burst: 3 tx on this card in the last 5 minutes, 2 confirmed fraud", "weight": 0.62 },
    { "factor": "device linked to 15 cards, 5 with prior fraud (3.0x baseline)", "weight": 0.28 } ] }
```

Reasons are deterministic (<1 ms), computed from training-period statistics only (no
future leakage), and distinguish **risky** from **protective** signals. On confirmed test
fraud, ~80% receive a substantive (non-fallback) reason.

---

## Run it locally

**Prerequisites:** Docker Desktop, a Python 3.11+ venv, and the Kaggle-produced artifacts in
`artifacts/` + `data/` (see *Reproduce* below).

```bash
pip install -r requirements-serve.txt
docker compose up -d                        # Kafka, Redis, Prometheus, Grafana
uvicorn src.serving.api:app --port 8000     # scoring API

# synchronous scoring
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"TransactionAmt":59,"ProductCD":"W","card1":7919,"addr1":204,"TransactionDT":8000000,"TransactionID":1}'

# load test (latency percentiles + decision breakdown)
python -m src.loadtest --n 100

# streaming path (two terminals)
python -m src.serving.consumer          # terminal A
python -m src.serving.producer --n 100  # terminal B
```

- **Grafana dashboard:** http://localhost:3000 (anonymous, no login) → *FraudGraph — Real-time Monitoring*
- **Prometheus:** http://localhost:9090

Five metrics are instrumented: `fraud_score` histogram, `decision_total` counter,
`inference_latency_seconds` histogram, `transaction_amount_mean` (5-min rolling), and
`feature_drift_alert` (fires when the rolling amount mean drifts >2σ from the training mean).

### Reproduce the artifacts (Kaggle)

Run the notebooks in order, downloading each stage's outputs into `artifacts/` (and `data/`
for the serving sample):
`01_eda_baseline` → `02_graph_construction` → `03_graphsage_train` (GPU) →
`04_explainability` → `05_serving_sample`.

---

## Honest caveats

- **Serving feature gap.** A live `/score` payload carries 6 core fields; the model was
  trained on ~430. The demo median-imputes the rest (a realistic feature-store cold-start),
  which compresses live scores — the graph embeddings still carry entity signal. The
  enriched replay (`test_sample_full.csv`) injects the materialized vectors and restores
  full discrimination.
- **Single temporal split**, not k-fold — random CV leaks the future on time-ordered data.
- **Categorical vocabulary** was built over the full dataset (a labeling convenience that
  leaks no outcome/future information; a stricter build would freeze it on train only).

---

## What I'd do differently at production scale

- **Distributed graph store** (Amazon Neptune / feature-store-backed graph) instead of a
  static `graph.pt`, so the graph grows continuously from the live stream.
- **Feature store** to serve the full feature vector at request time, closing the gap above.
- **Drift-triggered retraining**: `feature_drift_alert` becomes the trigger for an automated
  retrain + shadow-eval pipeline instead of a fixed schedule.
- **A/B / shadow deployment** for model versions, plus a **blocklist/allowlist pre-filter**
  and a short-TTL score cache to skip the model on known entities.

---

## Stack

Python · PyTorch Geometric · XGBoost · MLflow · FastAPI · Apache Kafka · Redis ·
Prometheus · Grafana · Docker Compose
