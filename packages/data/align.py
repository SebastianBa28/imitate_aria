import matplotlib.pyplot as plt
import numpy as np

from load_dataset import TRAY, clean, load_table, prepare_arrays, split_episodes

ANCHOR = 8


def first_hand(tips: np.ndarray) -> np.ndarray:
    """(5,3) fingertips from the first frame where all five are valid."""
    valid = np.isfinite(tips).all(axis=(1, 2))
    return tips[np.argmax(valid)]


def centroid(points: np.ndarray) -> np.ndarray:
    """(N,3) -> (3,) mean point, ignoring NaN."""
    return np.nanmean(points, axis=0)

def yaw_rotation(f_w: np.ndarray, f_s: np.ndarray) -> np.ndarray:
    """(5,3), (5,3) -> (3,3) rotation about z that turns f_w onto f_s."""
    a = f_w[:, :2] - centroid(f_w[:, :2])
    b = f_s[:, :2] - centroid(f_s[:, :2])
    wx, wy = a.T                          
    sx, sy = b.T

    cross = np.sum(wx * sy - wy * sx)
    dot = np.sum(wx * sx + wy * sy)
    theta = np.arctan2(cross, dot)        
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

def align_episode(e: dict, anchor: dict) -> dict:
    """Return {"tips": (T,5,3), "obj": (T,700,3)} in the anchor's frame.

    Position comes from the tray only (the paper uses all object points).
    Yaw comes from the hand. Both are applied to everything.
    """
    c_w = centroid(e["obj"][0, TRAY])
    c_s = centroid(anchor["obj"][0, TRAY])

    Rz = yaw_rotation(first_hand(e["tips"]), first_hand(anchor["tips"]))

    tips = (e["tips"] - c_w) @ Rz.T + c_s
    obj = (e["obj"] - c_w) @ Rz.T + c_s
    return {"tips": tips, "obj": obj}


def test_yaw() -> None:
    f = np.random.default_rng(0).random((5, 3))
    t = np.radians(30)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    assert np.allclose(yaw_rotation(f, f @ R.T + [1, 2, 3]), R), "yaw is wrong"
    print("yaw test passed")


def check(raw: dict, aligned: dict) -> None:
    target = centroid(aligned[ANCHOR]["obj"][0, TRAY])
    print("ep  centroid_err_cm  shape_err_cm  hand_err_cm")
    for i, e in aligned.items():
        c_err = np.linalg.norm(centroid(e["obj"][0, TRAY]) - target)
        # rotating and shifting must not change distances between fingers
        size = lambda t: np.linalg.norm(t - t[:, :1], axis=2)
        s_err = np.nanmax(np.abs(size(e["tips"]) - size(raw[i]["tips"])))
        h_err = np.linalg.norm(first_hand(e["tips"]) - first_hand(aligned[ANCHOR]["tips"]),
                               axis=1).mean()
        print(f"{i:>2}  {100*c_err:>15.2f}  {100*s_err:>12.2f}  {100*h_err:>11.1f}")


def show(episodes: dict, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, e in episodes.items():
        (line,) = ax.plot(*e["tips"][:, 1].T, label=f"ep {i}")
        ax.scatter(*e["obj"][0].T, s=1, color=line.get_color())
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()


def main() -> None:
    test_yaw()
    raw = clean(split_episodes(*prepare_arrays(load_table())))
    aligned = {i: align_episode(e, raw[ANCHOR]) for i, e in raw.items()}
    check(raw, aligned)
    show(raw, "before")
    show(aligned, "after")
    plt.show()


if __name__ == "__main__":
    main()