"""Time-series ring buffers for the dashboard plots (no ROS, no web: unit-tested on their own)."""
import threading
from collections import deque

import numpy as np


def _decimate(idx_count, max_points):
    """Indices of at most max_points samples spread evenly over idx_count, always keeping the last one."""
    if idx_count <= max_points:
        return np.arange(idx_count)
    return np.unique(np.linspace(0, idx_count - 1, max_points).round().astype(int))


class Series:
    """One signal: timestamps (s) and fixed-width float vectors. Thread-safe."""

    def __init__(self, width, history_s, labels=None, unit=""):
        self.width, self.history_s = width, history_s
        self.labels = list(labels) if labels else [str(i) for i in range(width)]
        self.unit = unit
        self._t, self._v = deque(), deque()
        self._lock = threading.Lock()

    def add(self, t, values):
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size != self.width:
            raise ValueError(f"expected {self.width} values, got {v.size}")
        with self._lock:
            self._t.append(float(t))
            self._v.append(v)
            while self._t and self._t[0] < t - self.history_s:
                self._t.popleft()
                self._v.popleft()

    def since(self, t_after, max_points):
        """Samples with t > t_after, decimated to max_points. Returns (t list, rows list of lists)."""
        with self._lock:
            ts = list(self._t)
            if not ts or ts[-1] <= t_after:
                return [], []
            # timestamps are appended in order: scan back from the end
            start = len(ts)
            while start > 0 and ts[start - 1] > t_after:
                start -= 1
            vs = list(self._v)[start:]
            ts = ts[start:]
        keep = _decimate(len(ts), max_points)
        return [ts[i] for i in keep], [_json_row(vs[i]) for i in keep]

    def last(self):
        with self._lock:
            return (self._t[-1], self._v[-1].copy()) if self._t else (None, None)


def _json_row(v):
    # JSON has no NaN: send null
    return [None if not np.isfinite(x) else round(float(x), 6) for x in v]


class SeriesStore:
    """Named series, created on first use for scalar metrics."""

    def __init__(self, history_s):
        self.history_s = history_s
        self.series = {}
        self._lock = threading.Lock()

    def define(self, name, width, labels=None, unit=""):
        with self._lock:
            self.series[name] = Series(width, self.history_s, labels, unit)
        return self.series[name]

    def add(self, name, t, values):
        s = self.series.get(name)
        if s is None:  # scalar metric seen for the first time
            with self._lock:
                s = self.series.setdefault(name, Series(1, self.history_s, [name.rsplit(".", 1)[-1]]))
        s.add(t, values)

    def describe(self):
        with self._lock:
            return {k: {"width": s.width, "labels": s.labels, "unit": s.unit} for k, s in self.series.items()}

    def since(self, cursors, max_points):
        """Incremental update: {name: {"t": [...], "v": [[...]]}} for samples newer than cursors[name].

        Updates cursors in place.
        """
        out = {}
        with self._lock:
            items = list(self.series.items())
        for name, s in items:
            t, v = s.since(cursors.get(name, float("-inf")), max_points)
            if t:
                out[name] = {"t": t, "v": v}
                cursors[name] = t[-1]
        return out
