"""Per-recording health stats from a post_process _aina.npz -> one JSON row."""
import json, sys, numpy as np

def runs(mask):
    r, c = [], 0
    for m in mask:
        if m: c += 1
        elif c: r.append(c); c = 0
    if c: r.append(c)
    return r

def stats(path):
    d = np.load(path, allow_pickle=True)
    T = len(d['rgb_ts']); fps = float(d['fps'])
    out = dict(rec=str(d['vrs_path']).split('/')[-1][:-4], frames=T, dur_s=round(T / fps, 2), start=int(d['start']),
               pose_pct=round(100 * float(np.isfinite(d['T_world_device'][:, 0, 0]).mean()), 1),
               sessions=int(len(np.unique(d['session_idx'][d['session_idx'] >= 0]))),
               no_pose_frames=[int(i) for i in np.where(~np.isfinite(d['T_world_device'][:, 0, 0]))[0]])
    lm = d['hand_landmarks_world']
    for h, name in enumerate(['left', 'right']):
        ok = np.isfinite(lm[:, h]).all(axis=(1, 2)); idx = np.where(ok)[0]
        mid = runs(~ok[idx[0]:idx[-1] + 1]) if len(idx) else []
        out[f'{name}_pct'] = round(100 * float(ok.mean()), 1)
        out[f'{name}_gaps'] = len(mid); out[f'{name}_longest_gap'] = max(mid, default=0)
        if name == 'right' and len(idx) > 1:
            v = np.linalg.norm(np.diff(lm[:, h, :5], axis=0), axis=-1) * fps
            vv = v[ok[1:] & ok[:-1]]
            out['right_tip_speed_max'] = round(float(np.nanmax(vv)), 2) if len(vv) else None
            out['right_jumps_gt2ms'] = int((vv.max(1) > 2).sum()) if len(vv) else None
    vis, pw, oid = d['vis'].astype(bool), d['points_world'], d['obj_id']
    for k, n in enumerate(d['object_names']):
        m = oid == k; has3d = np.isfinite(pw[:, m, 0]); key = str(n).split()[-1]
        out[f'{key}_vis_pct'] = round(100 * float(vis[:, m].mean()), 1)
        out[f'{key}_3d_pct'] = round(100 * float(has3d.mean()), 1)
        pose = np.isfinite(d['T_world_device'][:, 0, 0])[:, None]; vp = vis[:, m] & pose
        out[f'{key}_3d_given_vis_pct'] = round(100 * float((has3d & vp).sum() / max(vp.sum(), 1)), 1)
        out[f'{key}_bad_frames'] = int((has3d.mean(1) < 0.2).sum())
        # strays: points whose distance to the object's median changed >5 cm between the first and last frame
        # with 3D (the object is treated as rigid, so a point that left it shows up here)
        both = has3d[0] & has3d[-1]
        if both.sum() > 3:
            p0, p1 = pw[0, m][both], pw[-1, m][both]
            r0 = np.linalg.norm(p0 - np.median(p0, 0), axis=1); r1 = np.linalg.norm(p1 - np.median(p1, 0), axis=1)
            out[f'{key}_strays'] = int((np.abs(r1 - r0) > 0.05).sum())
        else:
            out[f'{key}_strays'] = None
        out[f'{key}_depth_m'] = round(float(np.nanmedian(d['depth'][:, m])), 2)
    return out

if __name__ == '__main__':
    print(json.dumps(stats(sys.argv[1])))
