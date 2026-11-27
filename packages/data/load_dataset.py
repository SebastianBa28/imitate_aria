"""Load imitate-aria and export one episode for the visualizer."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "SebastianBa28/imitate-aria"
FPS = 10
BAD = {2, 3, 4, 5, 6, 14}
# object point layout, from the dataset's meta/objects.json
CUBE = slice(0, 250)     # "rubiks cube": points 0 to 249
TRAY = slice(250, 700)   # "laptop" (our tray): points 250 to 699
MAX_GAP = 5   # gaps shorter than this many frames get filled



def load_table() -> pa.Table:
    root = Path(snapshot_download(REPO_ID, repo_type="dataset",
                                  allow_patterns=["meta/*", "data/*"]))
    files = sorted(root.glob("data/**/*.parquet"))
    return pa.concat_tables([pq.read_table(f) for f in files])


def col(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def inspect(table: pa.Table) -> None:
    print("rows:", table.num_rows)
    for name in table.column_names:
        print(f"{name:35s} {col(table, name).shape}")
    print("first state:\n", col(table, "observation.state")[0].reshape(5, 3))


def prepare_arrays(table: pa.Table):
    """separate the extracted data, set invalid observations to np.nan, 
    reshape the data of the tips to have the coordinates for each of the 5 tips at each frame"""
    ep = col(table, "episode_index").astype(int)       # (637,)
    frame = col(table, "frame_index").astype(int)      # (637,)

    tips = col(table, "observation.state").reshape(-1, 5, 3)
    tips_ok = col(table, "observation.state_valid") > 0.5       # True/False

    obj = col(table, "observation.object_points")               # (637, 700, 3)
    obj_ok = col(table, "observation.object_points_valid") > 0.5

    tips[~tips_ok] = np.nan
    obj[~obj_ok] = np.nan

    return ep, frame, tips, obj

def split_episodes(ep, frame, tips, obj) -> dict:
    '''Return {episode: {"tips": (Ti,5,3), "obj": (Ti,700,3)}.'''
    data = {}
    for e in np.unique(ep):                      
        rows = np.where(ep == e)[0]              
        rows = rows[np.argsort(frame[rows])]     
        data[int(e)] = {"tips": tips[rows], "obj": obj[rows]}
    return data


def report(episodes: dict) -> None:
    """Print one line per episode and flag bad tracking."""
    print("ep  frames  hand%  far%  high%  verdict")
    for i, e in episodes.items():
        tips, obj = e["tips"], e["obj"]
        hand_ok = np.isfinite(tips).all(axis=(1, 2))

        # first stable frame: look for blinks in the first 20 frames
        count = np.isfinite(obj).all(axis=2).sum(axis=1)
                # first stable frame: look for blinks in the first 20 frames
        count = np.isfinite(obj).all(axis=2).sum(axis=1)
        empty = np.where(count[:5] == 0)[0]
        start = empty[-1] + 1 if len(empty) else 0
        # wrong object: points far from, or well above, the hand
        pts = obj[start][np.isfinite(obj[start]).all(axis=1)]
        hand = tips[hand_ok].reshape(-1, 3)
        far = high = 1.0
        if len(pts) and len(hand):
            dist = np.linalg.norm(pts[:, None] - hand[None], axis=2).min(axis=1)
            far = (dist > 0.50).mean()
            high = (pts[:, 2] > np.median(hand[:, 2]) + 0.25).mean()

        flags = []
        if far > 0.2 or high > 0.1:
            flags.append("WRONG OBJECT?")
        if hand_ok.mean() < 0.8:
            flags.append("hand gaps")
        if start > 0:
            flags.append(f"trim to {start}")

        print(f"{i:>2}  {len(tips):>6}  {100*hand_ok.mean():>5.0f}  {100*far:>4.0f}  "
              f"{100*high:>5.0f}  {', '.join(flags) or 'OK'}")

def forward_fill(obj):
    """Replace missing points with their last valid position."""
    obj = obj.copy()
    for t in range(1, len(obj)):
        gap = ~np.isfinite(obj[t]).all(axis=1)
        obj[t][gap] = obj[t - 1][gap]
    return obj

def fill_short_gaps(tips: np.ndarray) -> np.ndarray:
    """(T,5,3) -> same, short hand gaps filled with a straight line."""
    tips = tips.copy()
    ok = np.isfinite(tips).all(axis=(1, 2))   # mask: (T,) True where the hand is valid
    T = len(tips)
    t = 0
    while t < T:
        if ok[t]:
            t += 1
            continue
        start = t                             # first NaN frame
        while t < T and not ok[t]:            # count the NaN frames
            t += 1
        end = t                               # first valid frame after the gap
        length = end - start
        # interpolate only if short and there is a valid frame on both sides
        if length < MAX_GAP and start > 0 and end < T:
            before, after = tips[start - 1], tips[end]
            for k in range(start, end):
                w = (k - start + 1) / (length + 1)    # 0 = before, 1 = after
                tips[k] = (1 - w) * before + w * after
    return tips

def clean(episodes: dict) -> dict:
    out = {}
    for i, e in episodes.items():
        if i in BAD:
            continue
        out[i] = {"tips": fill_short_gaps(e["tips"][2:]),
                  "obj": forward_fill(e["obj"][2:])}
    return out



def main() -> None:
    episodes = split_episodes(*prepare_arrays(load_table()))
    report(episodes)                       # check the raw data

    episodes = clean(episodes)             # drop bad demos, trim, fill gaps
    total = sum(len(e["tips"]) for e in episodes.values())
    print("kept:", sorted(episodes), "frames:", total)


if __name__ == "__main__":
    main()