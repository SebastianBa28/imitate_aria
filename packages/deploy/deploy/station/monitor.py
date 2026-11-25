"""Per-topic health: receive rate and age against an expected rate."""
import threading
import time
from collections import deque


class TopicMonitor:
    def __init__(self, expected_hz=None, window_s=2.0):
        self.expected_hz = expected_hz
        self.window_s = window_s
        self._times = deque(maxlen=2000)
        self._lock = threading.Lock()

    def tick(self, t=None):
        with self._lock:
            self._times.append(time.time() if t is None else t)

    def status(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            times = [x for x in self._times if x >= now - self.window_s]
            last = self._times[-1] if self._times else None
        if last is None:
            return {"status": "missing", "hz": None, "age_s": None}
        age = now - last
        hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 and times[-1] > times[0] else 0.0
        if self.expected_hz:
            stale_after = max(3.0 / self.expected_hz, 0.5)
            if age > stale_after:
                state = "stale"
            elif hz < 0.7 * self.expected_hz:
                state = "slow"
            else:
                state = "ok"
        else:  # command topics: only informative, they are quiet while idle
            state = "ok" if age < 1.0 else "idle"
        return {"status": state, "hz": round(hz, 1), "age_s": round(age, 3)}


class HealthBoard:
    def __init__(self, expected):
        self.monitors = {topic: TopicMonitor(hz) for topic, hz in expected.items()}

    def watch(self, topic, expected_hz=None):
        return self.monitors.setdefault(topic, TopicMonitor(expected_hz))

    def tick(self, topic, t=None):
        self.watch(topic).tick(t)

    def snapshot(self, now=None):
        return {topic: m.status(now) | {"expected_hz": m.expected_hz} for topic, m in self.monitors.items()}
