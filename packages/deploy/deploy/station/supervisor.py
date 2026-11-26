"""Starts, stops and watches the station's processes for the dashboard. No ROS imports, so
tests/test_supervisor.py runs without ROS.

- Each process gets its own session (= setsid) and process group: stop signals the whole group with
  SIGINT, then SIGTERM after stop_timeout_s, then SIGKILL after another stop_timeout_s.
- Processes run through deploy.station.child, which makes the kernel stop them if the station dies.
  For that to work they must be spawned from one thread that lives as long as the station, so all
  spawns go through the supervisor's own spawner thread.
- An unexpected exit is reported (on_event + state "exited" with its code) and never restarted
  automatically: a crash stays visible until someone acts on it.
- stdout and stderr are kept per process, the last log_lines lines, across restarts.
"""
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

from deploy.station.child import PARENT_ENV

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MAX_LINE = 2000


@dataclass
class Spec:
    argv: list
    title: str = ""
    cwd: str | None = None
    env: dict | None = None      # added to the station's environment


class _Proc:
    def __init__(self, name, spec, log_lines):
        self.name, self.spec = name, spec
        self.popen = None
        self.state = "stopped"   # stopped | running | stopping | exited (on its own)
        self.returncode = None
        self.started_at = self.ended_at = None
        self.log = deque(maxlen=log_lines)   # (id, t, text)
        self.next_id = 0
        self.op_lock = threading.Lock()      # one start/stop at a time per process


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class Supervisor:
    def __init__(self, specs, *, log_lines=500, stop_timeout_s=5.0, on_event=None, guard=None, wrap=True):
        """specs: {name: Spec}, in start order. guard(name) returns a reason to refuse a start, or None.
        on_event(level, text) is called for starts, stops and unexpected exits."""
        self.procs = {name: _Proc(name, spec, log_lines) for name, spec in specs.items()}
        self.stop_timeout_s = stop_timeout_s
        self.on_event = on_event or (lambda level, text: None)
        self.guard = guard or (lambda name: None)
        self.wrap = wrap
        self._lock = threading.Lock()        # states and logs
        self._closed = False
        self._spawn_q = queue.Queue()
        threading.Thread(target=self._spawner, daemon=True, name="supervisor-spawner").start()

    # --- commands: each returns (ok, message) ----------------------------------------------
    def start(self, name):
        p = self._get(name)
        with p.op_lock:
            if self._closed:
                return False, "the station is shutting down"
            self._poll(p)
            if p.state in ("running", "stopping"):
                return False, f"{name} is already {p.state}"
            reason = self.guard(name)
            if reason:
                return False, f"{name}: {reason}"
            fut = Future()
            self._spawn_q.put((p, fut))
            try:
                popen = fut.result(timeout=10)
            except Exception as e:  # noqa: BLE001  (bad argv, missing cwd, ...)
                self._log(p, f"--- failed to start: {e} ---")
                self.on_event("error", f"{name} failed to start: {e}")
                return False, f"{name} failed to start: {e}"
            with self._lock:
                p.popen, p.state, p.returncode = popen, "running", None
                p.started_at, p.ended_at = time.time(), None
            self._log(p, f"--- started, pid {popen.pid} ---")
            threading.Thread(target=self._reader, args=(p, popen), daemon=True, name=f"log-{name}").start()
        self.on_event("info", f"{name} started (pid {popen.pid})")
        return True, f"{name} started (pid {popen.pid})"

    def stop(self, name):
        p = self._get(name)
        self._poll(p)
        if p.state != "running":
            return False, f"{name} is not running"
        return self._terminate([p])[0]

    def restart(self, name):
        p = self._get(name)
        self._poll(p)
        if p.state == "running":
            ok, msg = self._terminate([p])[0]
            if not ok:
                return ok, msg
        return self.start(name)

    def start_all(self):
        results = [self.start(p.name) for p in self.procs.values() if self._poll(p) not in ("running", "stopping")]
        if not results:
            return True, "everything is already running"
        return all(ok for ok, _ in results), "; ".join(msg for _, msg in results)

    def stop_all(self):
        running = [p for p in self.procs.values() if self._poll(p) == "running"]
        if not running:
            return True, "nothing is running"
        results = self._terminate(running)
        return all(ok for ok, _ in results), "; ".join(msg for _, msg in results)

    def shutdown(self):
        """Stops everything and refuses further starts (station exit)."""
        self._closed = True
        return self.stop_all()

    # --- state -------------------------------------------------------------------------------
    def status(self):
        now = time.time()
        out = []
        for p in self.procs.values():
            self._poll(p)
            with self._lock:
                alive = p.state in ("running", "stopping")
                out.append({
                    "name": p.name, "title": p.spec.title, "state": p.state,
                    "pid": p.popen.pid if alive else None,
                    "returncode": p.returncode,
                    "started_at": p.started_at, "ended_at": p.ended_at,
                    "uptime_s": round(now - p.started_at, 1) if alive else None,
                })
        return out

    def logs_since(self, cursors, limit=200):
        """New log lines per process since cursors ({name: last id}, updated in place):
        {name: [[id, t, text], ...]}, at most `limit` (the newest) per process."""
        out = {}
        with self._lock:
            for p in self.procs.values():
                last = cursors.get(p.name, -1)
                lines = [list(x) for x in p.log if x[0] > last][-limit:]
                if lines:
                    out[p.name] = lines
                    cursors[p.name] = lines[-1][0]
        return out

    # --- internals ---------------------------------------------------------------------------
    def _get(self, name):
        if name not in self.procs:
            raise ValueError(f"unknown process {name!r}; known: {', '.join(self.procs)}")
        return self.procs[name]

    def _log(self, p, text):
        with self._lock:
            p.log.append((p.next_id, time.time(), text))
            p.next_id += 1

    def _spawner(self):
        while True:
            p, fut = self._spawn_q.get()
            try:
                fut.set_result(self._spawn(p.spec))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def _spawn(self, spec):
        argv = list(spec.argv)
        if self.wrap:
            argv = [sys.executable, "-m", "deploy.station.child", *argv]
        env = os.environ.copy()
        env.update(spec.env or {})
        env.update({"PYTHONUNBUFFERED": "1", "RCUTILS_COLORIZED_OUTPUT": "0", PARENT_ENV: str(os.getpid())})
        return subprocess.Popen(argv, cwd=spec.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)

    def _reader(self, p, popen):
        for raw in popen.stdout:
            self._log(p, ANSI.sub("", raw.decode(errors="replace").rstrip("\r\n"))[:MAX_LINE])
        self._reaped(p, popen, popen.wait())

    def _poll(self, p):
        """Notices an exit even while a leftover process still holds the log pipe open. Returns the state."""
        popen = p.popen
        if popen is not None and p.state in ("running", "stopping"):
            code = popen.poll()
            if code is not None:
                self._reaped(p, popen, code)
        return p.state

    def _reaped(self, p, popen, code):
        _killpg(popen.pid, signal.SIGKILL)   # anything it left behind in its group
        with self._lock:
            if p.popen is not popen or p.state not in ("running", "stopping"):
                return   # already handled
            expected = p.state == "stopping"
            p.state = "stopped" if expected else "exited"
            p.returncode, p.ended_at = code, time.time()
            last = next((t for _, _, t in reversed(p.log) if t and not t.startswith("---")), "")
        self._log(p, f"--- {'stopped' if expected else 'exited'} (code {code}) ---")
        if not expected:
            tail = f": {last}" if last else ""
            self.on_event("error" if code else "warn", f"{p.name} exited on its own (code {code}){tail}")

    def _terminate(self, procs):
        """SIGINT -> SIGTERM -> SIGKILL to each process group, all at once. One (ok, msg) per process."""
        locks = [p.op_lock for p in procs]
        for lock in locks:
            lock.acquire()
        try:
            todo = []
            with self._lock:
                for p in procs:
                    if p.state == "running":
                        p.state = "stopping"
                        todo.append(p)
            for p in todo:
                self._log(p, "--- stopping (SIGINT) ---")
            escalation = [None, signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
            how = {}
            for i, sig in enumerate(escalation[1:], start=1):
                waiting = [p for p in todo if p.popen.poll() is None]
                for p in waiting:
                    _killpg(p.popen.pid, sig)
                    how[p.name] = sig.name
                    if i > 1:
                        self._log(p, f"--- still running, sent {sig.name} ---")
                deadline = time.monotonic() + self.stop_timeout_s
                for p in waiting:
                    try:
                        p.popen.wait(timeout=max(0.0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        pass
                if all(p.popen.poll() is not None for p in todo):
                    break
            results = []
            for p in procs:
                if p not in todo:
                    results.append((False, f"{p.name} is not running"))
                    continue
                code = p.popen.poll()
                if code is None:   # survived SIGKILL: should not happen (uninterruptible sleep)
                    results.append((False, f"{p.name} (pid {p.popen.pid}) did not stop"))
                    continue
                self._reaped(p, p.popen, code)
                note = "" if how.get(p.name) == "SIGINT" else f" after {how.get(p.name)}"
                results.append((True, f"{p.name} stopped{note}"))
                self.on_event("info" if not note else "warn", f"{p.name} stopped{note}")
            return results
        finally:
            for lock in reversed(locks):
                lock.release()
