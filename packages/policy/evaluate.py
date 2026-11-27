from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from dataset import load_aligned, make_samples
from model import Policy

OUT = Path(__file__).parent / "policies"
MODEL = "policy_best.pt"   # which trained policy to load
EP = 13                      # a validation episode (never trained on)

tips, obj, target, ep = make_samples(load_aligned())
sel = np.where(ep == EP)[0]                  # sample numbers from this episode

model = Policy()
model.load_state_dict(torch.load(OUT / MODEL, map_location="cpu"))
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(tips[sel]).float(),
                 torch.from_numpy(obj[sel]).float()).numpy()       # (n,30,5,3)

err = np.sqrt(((pred - target[sel]) ** 2).mean()) * 100
print(f"episode {EP}: {len(sel)} samples, error {err:.2f} cm")

# fixed axis limits for the whole episode, so the view doesn't jump
pts = np.concatenate([obj[sel, -1].reshape(-1, 3), target[sel].reshape(-1, 3)])
lo, hi = pts.min(axis=0), pts.max(axis=0)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")


def draw(j):
    """Draw moment j of the episode."""
    ax.cla()                                              # clear the previous frame
    s = sel[j]
    ax.scatter(*obj[s, -1].T, s=1, c="lightgray")         # tray + cube now
    for f in range(5):                                    # all 5 fingers
        first = f == 0                                    # label only once for the legend
        ax.plot(*tips[s, :, f].T, "k-", lw=1, label="past 1 s" if first else None)
        ax.plot(*target[s, :, f].T, "g-", lw=1, label="real next 3 s" if first else None)
        ax.plot(*pred[j, :, f].T, "r--", lw=1, label="predicted" if first else None)
    ax.scatter(*tips[s, -1].T, c="k", s=15)               # fingertips right now
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)                            # equal scale on all axes
    ax.set_title(f"episode {EP}, moment {j + 1} of {len(sel)}")
    ax.legend(loc="upper left")


anim = FuncAnimation(fig, draw, frames=len(sel), interval=300)   # 300 ms per frame
plt.show()