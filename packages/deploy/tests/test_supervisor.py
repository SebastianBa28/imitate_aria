"""Process supervisor and component list: no ROS needed (fake children are small Python scripts)."""
import subprocess
import sys
import textwrap
import time

import pytest

from deploy.components import ROBOT_DOMAIN_ID, components, domain_refusal, is_live
from deploy.station.supervisor import Spec, Supervisor


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def gone(pid):
    """True once pid no longer exists (a zombie waiting for init counts as gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):   # exited between open and read
        return True


def lines(sv, name):
    return [t for _, _, t in sv.procs[name].log]


def state(sv, name):
    return next(p for p in sv.status() if p["name"] == name)


SLEEPER = py("""
    import time
    print("hello", flush=True)
    time.sleep(60)
""")


@pytest.fixture
def make():
    made = []

    def _make(specs, **kw):
        events = []
        sv = Supervisor(specs, on_event=lambda level, text: events.append((level, text)), **kw)
        sv.events = events
        made.append(sv)
        return sv
    yield _make
    for sv in made:
        sv.shutdown()


def test_start_captures_output_and_stop_ends_it(make):
    sv = make({"a": Spec(SLEEPER, title="A")})
    ok, msg = sv.start("a")
    assert ok, msg
    s = state(sv, "a")
    assert s["state"] == "running" and s["pid"] and s["title"] == "A"
    assert wait_for(lambda: "hello" in lines(sv, "a"))
    pid = s["pid"]
    ok, msg = sv.stop("a")
    assert ok and msg == "a stopped", msg
    s = state(sv, "a")
    assert s["state"] == "stopped" and s["pid"] is None
    assert wait_for(lambda: gone(pid))
    assert ("info", "a stopped") in sv.events


def test_crash_is_reported_with_code_and_last_line_and_not_restarted(make):
    sv = make({"c": Spec(py("print('boom', flush=True); raise SystemExit(3)"))})
    assert sv.start("c")[0]
    assert wait_for(lambda: state(sv, "c")["state"] == "exited")
    s = state(sv, "c")
    assert s["returncode"] == 3 and s["pid"] is None
    assert ("error", "c exited on its own (code 3): boom") in sv.events
    time.sleep(0.3)
    assert state(sv, "c")["state"] == "exited"   # no automatic restart


def test_refusals(make):
    blocked = {"b"}
    sv = make({"a": Spec(SLEEPER), "b": Spec(SLEEPER)}, guard=lambda n: "nope" if n in blocked else None)
    assert sv.start("a")[0]
    assert sv.start("a") == (False, "a is already running")
    assert sv.start("b") == (False, "b: nope")
    assert state(sv, "b")["state"] == "stopped"
    assert sv.stop("b") == (False, "b is not running")
    with pytest.raises(ValueError):
        sv.start("zzz")


def test_failed_spawn_is_reported(make):
    sv = make({"x": Spec(["/nonexistent/binary"])}, wrap=False)
    ok, msg = sv.start("x")
    assert not ok and "failed to start" in msg
    assert state(sv, "x")["state"] == "stopped"


def test_stop_escalates_when_sigint_is_ignored(make):
    stubborn = py("""
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(60)
    """)
    sv = make({"s": Spec(stubborn)}, stop_timeout_s=0.3)
    assert sv.start("s")[0]
    assert wait_for(lambda: "ready" in lines(sv, "s"))
    ok, msg = sv.stop("s")
    assert ok and msg == "s stopped after SIGTERM", msg
    assert any("sent SIGTERM" in t for t in lines(sv, "s"))


def test_stop_kills_the_whole_process_group(make):
    parent = py("""
        import subprocess, time
        child = subprocess.Popen(["sleep", "60"])
        print("child", child.pid, flush=True)
        time.sleep(60)
    """)
    sv = make({"p": Spec(parent)})
    assert sv.start("p")[0]
    assert wait_for(lambda: any(t.startswith("child ") for t in lines(sv, "p")))
    grandchild = int(next(t for t in lines(sv, "p") if t.startswith("child ")).split()[1])
    assert sv.stop("p")[0]
    assert wait_for(lambda: gone(grandchild), timeout=3)


def test_restart_keeps_the_log_and_gets_a_new_pid(make):
    sv = make({"a": Spec(SLEEPER)})
    assert sv.start("a")[0]
    pid1 = state(sv, "a")["pid"]
    cursors = {}
    assert wait_for(lambda: "hello" in lines(sv, "a"))
    first = sv.logs_since(cursors)["a"]
    assert sv.logs_since(cursors) == {}   # incremental
    ok, msg = sv.restart("a")
    assert ok, msg
    pid2 = state(sv, "a")["pid"]
    assert pid2 != pid1
    assert wait_for(lambda: lines(sv, "a").count("hello") == 2)
    new = [t for _, _, t in sv.logs_since(cursors)["a"]]
    assert new[0] == "--- stopping (SIGINT) ---" and f"--- started, pid {pid2} ---" in new
    assert all(row[0] > first[-1][0] for row in sv.logs_since({"a": first[-1][0]})["a"])


def test_start_all_stop_all_and_shutdown(make):
    sv = make({"a": Spec(SLEEPER), "b": Spec(SLEEPER)})
    ok, msg = sv.start_all()
    assert ok, msg
    assert {p["state"] for p in sv.status()} == {"running"}
    assert sv.start_all() == (True, "everything is already running")
    ok, msg = sv.stop_all()
    assert ok and msg == "a stopped; b stopped", msg
    assert sv.stop_all() == (True, "nothing is running")
    assert sv.start("a")[0]
    pid = state(sv, "a")["pid"]
    assert sv.shutdown()[0]
    assert wait_for(lambda: gone(pid))
    assert sv.start("a") == (False, "the station is shutting down")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_children_die_when_the_station_is_killed(tmp_path):
    script = tmp_path / "station.py"
    script.write_text(textwrap.dedent("""
        import os, signal, sys, time
        from deploy.station.supervisor import Spec, Supervisor
        sv = Supervisor({"s": Spec(["sleep", "60"])})
        ok, msg = sv.start("s")
        print(sv.status()[0]["pid"], flush=True)
        time.sleep(0.5)   # let the child get past its parent check
        os.kill(os.getpid(), signal.SIGKILL)   # no cleanup at all
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=20)
    pid = int(out.stdout.split()[0])
    assert wait_for(lambda: gone(pid), timeout=5), f"child {pid} outlived the station; stderr: {out.stderr}"


# --- components ---------------------------------------------------------------------------
def test_mock_profile_components():
    comps = components("mock")
    assert [c.name for c in comps] == ["arm", "hand", "camera", "policy"]
    hand = comps[1]
    assert hand.label == "hand_node"
    assert hand.argv("py") == ["py", "-m", "deploy.nodes.hand_node", "--ros-args",
                               "-p", "mock:=true", "-p", "calibrate:=false"]
    assert comps[2].module == "deploy.nodes.mock_camera"
    assert [c.name for c in components("mock", policy=False)] == ["arm", "hand", "camera"]


def test_real_profile_components_and_overrides():
    comps = {c.name: c for c in components("real", hand="mock", calibrate=True)}
    assert "arm" not in comps   # the real arm runs on the NUC
    assert comps["hand"].params == ("mock:=true", "calibrate:=true")
    assert comps["camera"].module == "deploy.nodes.realsense_node"
    with pytest.raises(ValueError):
        components("sim")
    with pytest.raises(ValueError):
        components("mock", camera="fake")


def test_domain_refusal_and_liveness():
    assert domain_refusal("mock", ROBOT_DOMAIN_ID)
    assert domain_refusal("mock", "0") is None
    assert domain_refusal("real", ROBOT_DOMAIN_ID) is None
    assert is_live({"age_s": 0.1})
    assert not is_live({"age_s": 10.0})
    assert not is_live({"age_s": None})
    assert not is_live(None)
