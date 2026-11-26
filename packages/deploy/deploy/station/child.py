"""Exec wrapper for processes the supervisor starts: python -m deploy.station.child <argv...>

On Linux it asks the kernel to send SIGTERM to this process when the station dies (even by SIGKILL),
then replaces itself with <argv>, so no node keeps running without the dashboard that started it.
The supervisor spawns from one long-lived thread because the kernel tracks the spawning *thread*.
"""
import os
import signal
import sys

PARENT_ENV = "IMITATE_SUPERVISOR_PID"


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit("usage: python -m deploy.station.child <command> [args...]")
    if sys.platform.startswith("linux"):
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        expected = os.environ.get(PARENT_ENV)
        if expected and os.getppid() != int(expected):
            sys.exit("supervisor already gone")   # died before prctl took effect
    os.environ.pop(PARENT_ENV, None)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
