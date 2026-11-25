"""Station logic that needs neither ROS nor the web server."""
import math

import pytest

from deploy.station.buffers import Series, SeriesStore
from deploy.station.monitor import HealthBoard, TopicMonitor


def test_series_drops_old_samples_and_rejects_wrong_width():
    s = Series(2, history_s=1.05)  # not exactly 1.0: 0.1 steps aren't exact in floating point
    for i in range(30):
        s.add(i * 0.1, [i, -i])
    t, v = s.since(float("-inf"), 1000)
    assert t[0] == pytest.approx(1.9) and t[-1] == pytest.approx(2.9)
    assert v[-1] == [29.0, -29.0]
    with pytest.raises(ValueError):
        s.add(3.0, [1, 2, 3])


def test_series_since_is_incremental_and_decimated():
    s = Series(1, history_s=100)
    for i in range(100):
        s.add(float(i), [i])
    t, _ = s.since(89.0, 1000)
    assert t == [float(i) for i in range(90, 100)]
    t, _ = s.since(float("-inf"), 10)
    assert len(t) <= 10 and t[-1] == 99.0 and t[0] == 0.0


def test_nan_becomes_null():
    s = Series(2, history_s=10)
    s.add(0.0, [math.nan, 1.0])
    assert s.since(float("-inf"), 10)[1] == [[None, 1.0]]


def test_store_cursors_and_auto_scalars():
    st = SeriesStore(history_s=10)
    st.define("hand.q", 2, ["a", "b"], "rad")
    st.add("hand.q", 1.0, [0, 1])
    st.add("metric.latency_ms", 1.0, [12.0])
    cursors = {}
    first = st.since(cursors, 50)
    assert set(first) == {"hand.q", "metric.latency_ms"}
    assert st.since(cursors, 50) == {}
    st.add("hand.q", 2.0, [1, 2])
    assert st.since(cursors, 50)["hand.q"]["t"] == [2.0]
    assert st.describe()["metric.latency_ms"]["width"] == 1


def test_monitor_states():
    m = TopicMonitor(expected_hz=100)
    assert m.status(now=0)["status"] == "missing"
    for i in range(200):
        m.tick(i * 0.01)
    assert m.status(now=2.0)["status"] == "ok"
    assert m.status(now=2.0)["hz"] == pytest.approx(100, rel=0.02)
    assert m.status(now=3.0)["status"] == "stale"
    slow = TopicMonitor(expected_hz=100)
    for i in range(40):
        slow.tick(i * 0.05)  # 20 Hz
    assert slow.status(now=1.96)["status"] == "slow"


def test_command_topics_go_idle_not_stale():
    board = HealthBoard({"/x": 10})
    board.watch("/cmd")
    board.tick("/cmd", 0.0)
    snap = board.snapshot(now=5.0)
    assert snap["/cmd"]["status"] == "idle" and snap["/x"]["status"] == "missing"
