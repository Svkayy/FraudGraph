from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .inference import FraudScorer
from .redis_store import RedisStore

state = {}


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: float | int | str | None = None
    addr1: float | int | None = None
    DeviceInfo: str | None = None
    TransactionDT: float
    TransactionID: str | int | None = None
    # optional: pre-materialized feature vector from the feature store.
    # When absent, the scorer median-imputes the non-core fields.
    feature_vector: list[float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = FraudScorer()
    state["redis"] = RedisStore()
    yield
    state.clear()


app = FastAPI(title="FraudGraph Scoring API", lifespan=lifespan)


def score_transaction(payload: dict) -> dict:
    store: RedisStore = state["redis"]
    scorer: FraudScorer = state["scorer"]
    card1 = payload.get("card1")
    velocity = store.record_and_get_velocity(card1, float(payload["TransactionDT"]),
                                             float(payload["TransactionAmt"]))
    result = scorer.score(payload, velocity, feature_vector=payload.get("feature_vector"))
    if payload.get("TransactionID") is not None:
        store.push_neighbor(card1, payload["TransactionID"])
    return result


@app.get("/health")
def health():
    return {"status": "ok", "redis": state["redis"].ping() if "redis" in state else False}


@app.post("/score")
def score(txn: Transaction):
    return score_transaction(txn.model_dump())
