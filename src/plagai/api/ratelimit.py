"""In-process sliding-window rate limiter.

It is per process. When you run several replicas, move it to Redis
(INCR + EXPIRE) or enforce limits at the gateway.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[object, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: object) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
