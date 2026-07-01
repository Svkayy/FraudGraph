# FraudGraph — Real-Time Payment Fraud Detection with Graph Neural Networks

> A production-style fraud detection system that models payment transactions as a
> heterogeneous graph and serves fraud scores in real time using GraphSAGE over a
> Kafka + Redis streaming pipeline.

**Status:** 🚧 Under construction. This README is filled in fully at Stage 5.

## Problem

Traditional fraud detection scores each transaction in isolation — one row, one prediction.
Real fraud is relational: the same stolen card hits many merchants, one device fingerprint
spans many accounts, a ring of transactions clusters within minutes. FraudGraph builds the
transaction network as a graph and uses GraphSAGE to classify each transaction by aggregating
signals from its neighbors — catching coordinated fraud that tabular models miss.

## Architecture (draft)

```
Transaction ──► Kafka "transactions" ──► consumer ──► Redis (velocity cache)
                                            │
                                            ▼
                                     local subgraph ──► GraphSAGE ──► fraud score
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                        ▼
              Kafka "fraud-scores"                       FastAPI POST /score
                                                                 │
                                                    Prometheus ──┴──► Grafana
```

## Results

_Filled in after Stage 3 (XGBoost vs GraphSAGE AUC) and Stage 4 (latency)._

| Model      | AUC-ROC | Precision @ 10% recall |
|------------|---------|------------------------|
| XGBoost    | TBD     | TBD                    |
| GraphSAGE  | TBD     | TBD                    |

## Repository layout

- `notebooks/` — Kaggle notebooks for Stages 1–3 (EDA/baseline, graph, GraphSAGE).
- `src/` — shared model/feature code + local serving stack.
- `monitoring/` — Prometheus config + Grafana dashboard.
- `artifacts/`, `data/` — gitignored; populated by downloading from Kaggle.

## How to run locally

_Filled in at Stage 5._
