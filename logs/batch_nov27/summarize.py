"""stats.jsonl -> summary.md (batch medians, flagged recordings, per-recording table)."""
import json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(D, "stats.jsonl"))]
cols = ["rec", "frames", "start", "no_pose_frames", "right_pct", "right_longest_gap", "right_jumps_gt2ms", "left_pct",
        "cube_vis_pct", "cube_3d_given_vis_pct", "cube_strays", "box_vis_pct", "box_3d_given_vis_pct", "box_strays", "cube_depth_m"]
def agg(c):
    v = np.array([r[c] for r in rows if r.get(c) is not None], float)
    return f"median {np.median(v):g} (min {v.min():g}, max {v.max():g})"
summ = [f"{len(rows)} recordings, {sum(r['frames'] for r in rows)} frames, {sum(r['dur_s'] for r in rows):.0f} s at 10 fps; one VIO session each"]
summ += [f"{c}: {agg(c)}" for c in ["frames", "start", "pose_pct", "right_pct", "right_longest_gap", "right_tip_speed_max",
                                    "cube_vis_pct", "cube_3d_given_vis_pct", "cube_strays", "box_vis_pct", "box_3d_given_vis_pct", "box_strays"]]
flags = []
for r in rows:
    f = []
    if r['right_pct'] < 90: f.append(f"right hand tracked {r['right_pct']}%")
    if r['right_longest_gap'] >= 3: f.append(f"right-hand gap {r['right_longest_gap']} frames")
    if (r.get('right_jumps_gt2ms') or 0) > 0: f.append(f"{r['right_jumps_gt2ms']} fingertip jumps >2 m/s")
    if r['cube_3d_given_vis_pct'] < 90: f.append(f"cube depth only {r['cube_3d_given_vis_pct']}% of visible")
    if r['box_3d_given_vis_pct'] < 90: f.append(f"box depth only {r['box_3d_given_vis_pct']}% of visible")
    if (r.get('cube_strays') or 0) > 12: f.append(f"cube strays {r['cube_strays']}/250")
    if (r.get('box_strays') or 0) > 22: f.append(f"box strays {r['box_strays']}/450")
    if r['cube_vis_pct'] < 75: f.append(f"cube visible {r['cube_vis_pct']}%")
    if f: flags.append(f"- **{r['rec']}**: " + "; ".join(f))
tab = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
tab += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
md = ("# Batch health: nov27 recordings\n\n## Summary\n" + "\n".join("- " + s for s in summ) +
      "\n\n## Flagged (thresholds: right hand <90%, gap >=3 fr, depth <90% of visible points, strays >5% of points, cube visible <75%)\n" +
      ("\n".join(flags) or "none") + "\n\n## Per recording\n" + "\n".join(tab) +
      "\n\n_pct = % of point-frames. 3d_given_vis = visible points (in frames with a pose) that got a 3D position. "
      "strays = points whose distance to the object's median changed >5 cm between first and last frame (rigid-object check). "
      "no_pose_frames = frame indices with no VIO pose (NaN)._\n")
open(os.path.join(D, "summary.md"), "w").write(md)
print("\n".join(summ)); print(); print("\n".join(flags))
