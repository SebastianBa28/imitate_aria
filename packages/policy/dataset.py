import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from align import ANCHOR, align_episode
from load_dataset import clean, load_table, prepare_arrays, split_episodes

T_IN, T_OUT, N_POINTS = 10, 30, 500


def load_aligned() -> dict:
    raw = clean(split_episodes(*prepare_arrays(load_table())))
    return {i: align_episode(e, raw[ANCHOR]) for i, e in raw.items()}


def window_indices(t: int, T: int):
    """Return (past, future): 10 and 30 frame numbers, kept within 0..T-1."""
    Tp = [max(0,t-i) for i in range(9,-1,-1)]
    Tf = [min(T-1, t+i) for i in range(1,31)]
    return Tp, Tf


def pick_points(obj: np.ndarray, rng) -> np.ndarray:
    """(T,700,3) -> 500 point numbers, the same for the whole episode."""
    valid = np.isfinite(obj).all(axis=2)    
    good = valid.all(axis=0)                
    numbers = np.where(good)[0]             
    return rng.choice(numbers, N_POINTS, replace=len(numbers) < N_POINTS)


def make_samples(episodes: dict):
    tips_in, obj_in, target, ep_ids = [], [], [], []
    rng = np.random.default_rng(0)          # fixed seed: same choice every run
    for i, e in episodes.items():
        points = pick_points(e["obj"], rng)
        for t in range(len(e["tips"])):
            past, future = window_indices(t, len(e["tips"]))
            x = e["tips"][past]                 # (10,5,3)
            y = e["tips"][future]               # (30,5,3)
            o = e["obj"][past][:, points]       # (10,500,3)

            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            tips_in.append(x)
            obj_in.append(o)
            target.append(y)
            ep_ids.append(i)

    return tuple(np.stack(a) for a in (tips_in, obj_in, target, ep_ids))


if __name__ == "__main__":
    tips_in, obj_in, target, ep_ids = make_samples(load_aligned())
    print(tips_in.shape, obj_in.shape, target.shape, ep_ids.shape)
    assert np.isfinite(obj_in).all(), "NaN left in object points"