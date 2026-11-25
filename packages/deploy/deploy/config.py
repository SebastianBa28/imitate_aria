"""Loads config/station.yaml (or $IMITATE_DEPLOY_CONFIG). Relative paths resolve against packages/deploy/."""
import os
from pathlib import Path

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = DEPLOY_ROOT / "config" / "station.yaml"


def load(path=None):
    path = Path(path or os.environ.get("IMITATE_DEPLOY_CONFIG") or DEFAULT_CONFIG)
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    """Absolute path for a config entry given relative to packages/deploy/."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else DEPLOY_ROOT / p
