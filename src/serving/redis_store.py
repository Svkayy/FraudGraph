import time

import redis

from . import config


class RedisStore:
    def __init__(self, host=config.REDIS_HOST, port=config.REDIS_PORT):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except redis.exceptions.RedisError:
            return False

    def record_and_get_velocity(self, card1, ts: float, amount: float) -> dict:
        """Increment-then-read: add this transaction to the card's timeline,
        then count transactions in each rolling window (count INCLUDES the
        current one, matching the training-time rolling definition)."""
        zkey = f"card:{card1}:ts"
        seq = self.r.incr(f"card:{card1}:seq")
        member = f"{ts}:{seq}"

        pipe = self.r.pipeline()
        pipe.zadd(zkey, {member: ts})
        pipe.zremrangebyscore(zkey, "-inf", ts - config.WINDOWS["count_24h"])
        for field, win in config.WINDOWS.items():
            pipe.zcount(zkey, ts - win, ts)
        pipe.expire(zkey, config.VELOCITY_TTL_SEC)
        pipe.expire(f"card:{card1}:seq", config.VELOCITY_TTL_SEC)
        results = pipe.execute()

        counts = dict(zip(config.WINDOWS.keys(), results[2:5]))

        hkey = f"card:{card1}:velocity"
        self.r.hset(hkey, mapping={
            "count_1h": counts["count_1h"], "count_6h": counts["count_6h"],
            "count_24h": counts["count_24h"], "last_amount": amount, "last_timestamp": ts})
        self.r.expire(hkey, config.VELOCITY_TTL_SEC)
        return counts

    def push_neighbor(self, card1, transaction_id) -> None:
        lkey = f"card:{card1}:neighbors"
        pipe = self.r.pipeline()
        pipe.lpush(lkey, str(transaction_id))
        pipe.ltrim(lkey, 0, config.NEIGHBOR_LIST_LEN - 1)
        pipe.expire(lkey, config.VELOCITY_TTL_SEC)
        pipe.execute()

    def recent_neighbors(self, card1) -> list:
        return self.r.lrange(f"card:{card1}:neighbors", 0, config.NEIGHBOR_LIST_LEN - 1)
