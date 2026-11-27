import time

import numpy as np
import torch

from dataset import load_aligned, make_samples
from model import Policy

from pathlib import Path


OUT = Path(__file__).parent / "policies"          # packages/policy/policies

VAL = {0, 13}          # episodes held out for validation 
OVERFIT = False         # True: train on episode 8 only, no augmentation (sanity check)
EPOCHS = 2000          # paper value
BATCH = 32             # we can optimize it
LR = 1e-4              # learning rate also we can optimize it
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_torch(*arrays):
    """NumPy arrays -> float tensors on the GPU (or CPU)."""
    return [torch.from_numpy(a).float().to(device) for a in arrays]


def augment(tips, obj, target):
    """AINA's augmentation: one random yaw, scale and shift per sample, applied to
    input AND target; plus up to 2 cm noise on the input fingertips only."""
    B = tips.shape[0]
    c = obj[:, -1].mean(dim=1)[:, None, None, :]        # (B,1,1,3) centre of the scene now
    th = (torch.rand(B, device=device) * 2 - 1) * np.radians(60)
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0], R[:, 0, 1] = torch.cos(th), -torch.sin(th)
    R[:, 1, 0], R[:, 1, 1] = torch.sin(th), torch.cos(th)
    R[:, 2, 2] = 1
    s = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=device)             # scale 0.8 to 1.2
    shift = (torch.rand(B, 1, 1, 3, device=device) * 2 - 1) * 0.30    # +-30 cm

    def tf(x):                                           # (B,T,P,3) -> same
        return ((x - c) @ R.transpose(1, 2)[:, None]) * s + c + shift

    noise = (torch.rand_like(tips) * 2 - 1) * 0.02      # +-2 cm
    return tf(tips) + noise, tf(obj), tf(target)


def main():
    tips, obj, target, ep = make_samples(load_aligned())
    if OVERFIT:
        tr = va = ep == 8
    else:
        va = np.isin(ep, list(VAL))
        tr = ~va
    tr_x, tr_o, tr_y = to_torch(tips[tr], obj[tr], target[tr])
    va_x, va_o, va_y = to_torch(tips[va], obj[va], target[va])
    n = len(tr_x)
    print(f"device {device}, train {n}, val {len(va_x)}")

    model = Policy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best, start = float("inf"), time.time()

    for epoch in range(EPOCHS):
        model.train()                                    # training mode (dropout on)
        order = torch.randperm(n, device=device)         # shuffle every epoch
        total = 0.0
        for k in range(0, n, BATCH):
            idx = order[k:k + BATCH]
            x, o, y = tr_x[idx], tr_o[idx], tr_y[idx]
            if not OVERFIT:
                x, o, y = augment(x, o, y)
            loss = ((model(x, o) - y) ** 2).mean()       # MSE
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()                                     # evaluation mode (dropout off)
        with torch.no_grad():                            # no corrections needed here
            val = ((model(va_x, va_o) - va_y) ** 2).mean().item()
        if val < best:
            best = val
            torch.save(model.state_dict(), OUT/"policy_best.pt")
        if epoch % 20 == 0:
            # sqrt(MSE) in cm: roughly "how far off is a fingertip, on average"
            print(f"epoch {epoch:4d}  train {100*np.sqrt(total/n):6.2f} cm  "
                  f"val {100*np.sqrt(val):6.2f} cm  ({time.time()-start:.0f} s)")


if __name__ == "__main__":
    main()