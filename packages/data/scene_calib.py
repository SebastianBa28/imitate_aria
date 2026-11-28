#!/usr/bin/env python3
"""
Calibrate the fixed in-scene cameras (D435, OAK, ...) into one "world" frame from a ChArUco board clip.

world = TABLE frame: the board lying flat on the table at the start of the clip. Origin at the board's
top-left chessboard corner (as printed), x along the board's rows (to the right), y towards the top edge
of the board, z UP out of the table. Units: metres.

    # once: print packages/data/assets/charuco_board.pdf (A4 landscape, 100 % scale, no "fit to page"), glue it
    # flat onto something rigid, MEASURE one square and pass it below if it is not exactly 36 mm.
    # (re-create it with: uv run python packages/data/scene_calib.py --make-board)

    # every lab session (and whenever a camera may have moved): record the calibration clip with record_cams
    #   pixi run -e hw record-cams --name calib --data-dir <repo>/raw_data/scene
    # 1. board FLAT on the table in the workspace, both cameras see it, hold still >= 2 s (defines world)
    # 2. then pick it up and slowly tilt / move it through both views for ~10-15 s (refines camera<->camera)
    uv run python packages/data/scene_calib.py raw_data/scene/calib_nov30 [--square-mm 39.6]

Writes <clip>/calib.json and <clip>/calib_check.jpg (world axes + board corners drawn on each camera).
scene_process.py picks up the calibration clip of the same day automatically (or --calib PATH).

calib.json:
    world            text description of the frame
    board            board spec (squares, square/marker size, dictionary)
    cameras.<cam>    T_world_cam (4x4, camera->world), K (3x3), dist, size [W,H], rms_px, n_obs
    T_base_world     robot base <- world (4x4), null until the hand-eye calibration is done
    checks           cross-camera triangulation error of the board corners, camera baseline

Method: per camera, ChArUco corners -> solvePnP for every frame; T_world_cam from the still flat frames;
then one joint least-squares over all frames that both cameras see (camera poses + one board pose per
frame, the flat frames' board pose fixed at the world origin) minimizing corner reprojection error.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from aina.geometry import inv, transform, triangulate
from aina.scene_io import SceneRecording

BOARD = {"squares_x": 7, "squares_y": 5, "square_mm": 36.0, "marker_mm": 27.0, "dictionary": "DICT_5X5_100"}  # fits A4
T_WORLD_BOARD = np.diag([1.0, -1.0, -1.0, 1.0])  # board frame (z into the board) -> table frame (z up)
MIN_CORNERS = 6


def make_board(spec):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    return cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_mm"] / 1000.0,
                                  spec["marker_mm"] / 1000.0, d)


def write_board(spec, out, dpi):
    board = make_board(spec)
    px_per_mm = dpi / 25.4
    w_mm, h_mm = spec["squares_x"] * spec["square_mm"], spec["squares_y"] * spec["square_mm"]
    margin = int(round(5 * px_per_mm))
    img = board.generateImage((int(round(w_mm * px_per_mm)) + 2 * margin, int(round(h_mm * px_per_mm)) + 2 * margin),
                              marginSize=margin, borderBits=1)
    label = (f"ChArUco {spec['squares_x']}x{spec['squares_y']}  square {spec['square_mm']} mm  marker "
             f"{spec['marker_mm']} mm  {spec['dictionary']}  - print at 100 %, measure a square")
    img = np.vstack([img, np.full((int(12 * px_per_mm), img.shape[1]), 255, np.uint8)])
    cv2.putText(img, label, (margin, img.shape[0] - int(4 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX,
                px_per_mm * 0.35, 0, max(1, int(px_per_mm * 0.3)), cv2.LINE_AA)
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, "charuco_board.png")
    cv2.imwrite(png, img)
    from PIL import Image
    pdf = os.path.join(out, "charuco_board.pdf")
    Image.fromarray(img).save(pdf, resolution=dpi)
    with open(os.path.join(out, "charuco_board.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print(f"wrote {png} and {pdf} ({w_mm:.0f} x {h_mm:.0f} mm board + margin, {dpi} dpi)")


# ----------------------------------------------------------------------------- detection
def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def T_to_rt(T):
    return cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3].copy()


def detect(cam, board, stride):
    """{frame index: (ids (n,), corners (n,2), T_cam_board, rms px)} for frames with >= MIN_CORNERS corners."""
    det = cv2.aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()
    out = {}
    idxs = list(range(0, len(cam), stride))
    for i, rgb in zip(idxs, cam.colors(idxs)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = det.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            continue
        ids, corners = ids.reshape(-1), corners.reshape(-1, 2).astype(np.float64)
        obj = obj_all[ids].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, corners, cam.K, cam.dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, 1))))
        out[i] = (ids, corners, rt_to_T(rvec, tvec), rms)
    return out


def mean_pose(Ts):
    """Average of rigid transforms: rotation mean (quaternions), median translation."""
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.median(np.stack([T[:3, 3] for T in Ts]), 0)
    return out


# ----------------------------------------------------------------------------- joint refinement
def refine(cams, obs, T_world_cam0, T_world_board0, flat_frames, obj_all):
    """obs: list of (frame key, cam id, ids, corners). Optimizes T_world_cam per camera and T_world_board per
    non-flat frame (flat frames are fixed at T_WORLD_BOARD). Returns (T_world_cam, T_world_board, rms per cam)."""
    cam_ids = list(T_world_cam0)
    moving = sorted({k for k, *_ in obs if k not in flat_frames})
    ci = {c: i for i, c in enumerate(cam_ids)}
    fi = {k: i for i, k in enumerate(moving)}
    nc = 6 * len(cam_ids)

    def pack():
        x = []
        for c in cam_ids:
            x += list(np.r_[T_to_rt(inv(T_world_cam0[c]))])      # parametrize T_cam_world
        for k in moving:
            x += list(np.r_[T_to_rt(T_world_board0[k])])
        return np.array(x)

    def unpack(x):
        T_cw = {c: rt_to_T(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]) for c, i in ci.items()}
        T_wb = {k: rt_to_T(x[nc + 6 * j:nc + 6 * j + 3], x[nc + 6 * j + 3:nc + 6 * j + 6]) for k, j in fi.items()}
        return T_cw, T_wb

    def residuals(x):
        T_cw, T_wb = unpack(x)
        res = []
        for k, c, ids, uv in obs:
            T_cb = T_cw[c] @ T_wb.get(k, T_WORLD_BOARD)
            rvec, tvec = T_to_rt(T_cb)
            proj, _ = cv2.projectPoints(obj_all[ids], rvec, tvec, cams[c].K, cams[c].dist)
            res.append((proj.reshape(-1, 2) - uv).ravel())
        return np.concatenate(res)

    n_res = sum(2 * len(ids) for _, _, ids, _ in obs)
    sparsity = lil_matrix((n_res, nc + 6 * len(moving)), dtype=int)
    r = 0
    for k, c, ids, _ in obs:
        m = 2 * len(ids)
        sparsity[r:r + m, 6 * ci[c]:6 * ci[c] + 6] = 1
        if k in fi:
            sparsity[r:r + m, nc + 6 * fi[k]:nc + 6 * fi[k] + 6] = 1
        r += m
    sol = least_squares(residuals, pack(), jac_sparsity=sparsity, loss="huber", f_scale=1.5, x_scale="jac",
                        max_nfev=200)
    T_cw, T_wb = unpack(sol.x)
    res = residuals(sol.x).reshape(-1, 2)
    rms, r = {}, 0
    per = {c: [] for c in cam_ids}
    for k, c, ids, _ in obs:
        per[c].append(res[r:r + len(ids)])
        r += len(ids)
    for c in cam_ids:
        e = np.concatenate(per[c]) if per[c] else np.zeros((0, 2))
        rms[c] = float(np.sqrt(np.mean(np.sum(e ** 2, 1)))) if len(e) else float("nan")
    return {c: inv(T) for c, T in T_cw.items()}, T_wb, rms


def triangulation_check(cams, T_world_cam, pairs, dets, spec):
    """Triangulate the board corners seen by two cameras in the same (paired) frame; compare the distances of
    neighbouring corners with the square size. Independent of the board poses. Returns mean/max |error| mm."""
    c0, c1 = list(cams)[:2]
    P = {c: inv(T_world_cam[c])[:3] for c in (c0, c1)}   # normalized-coordinate projection matrices
    sq = spec["square_mm"] / 1000.0
    nx = spec["squares_x"] - 1                            # inner corners per row
    errs = []
    for i0, i1 in pairs:
        ids0, uv0 = dets[c0][i0][:2]
        ids1, uv1 = dets[c1][i1][:2]
        common = np.intersect1d(ids0, ids1)
        if len(common) < 4:
            continue
        pos0 = {int(k): j for j, k in enumerate(ids0)}
        pos1 = {int(k): j for j, k in enumerate(ids1)}
        n0 = cams[c0].undistort_px(uv0[[pos0[int(k)] for k in common]])
        n1 = cams[c1].undistort_px(uv1[[pos1[int(k)] for k in common]])
        X = {int(k): triangulate([P[c0], P[c1]], [a, b]) for k, a, b in zip(common, n0, n1)}
        for k, x in X.items():
            for nb in (k + 1, k + nx):                    # right and lower neighbours
                if nb in X and (nb != k + 1 or (k + 1) % nx):
                    errs.append(abs(np.linalg.norm(X[nb] - x) - sq))
    if not errs:
        return None
    errs = np.array(errs) * 1000
    return {"n": int(len(errs)), "mean_mm": round(float(errs.mean()), 2), "p95_mm": round(float(np.percentile(errs, 95)), 2)}


def draw_check(cam, rgb, T_world_cam, det, obj_all):
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    T_cw = inv(T_world_cam)
    rvec, tvec = T_to_rt(T_cw)
    cv2.drawFrameAxes(out, cam.K, cam.dist, rvec, tvec, 0.1, 3)
    if det is not None:
        ids, uv = det[:2]
        P_world = transform(T_WORLD_BOARD, obj_all[ids])
        proj = cam.project(transform(T_cw, P_world))
        for (u, v), (pu, pv) in zip(uv, proj):
            cv2.circle(out, (int(u), int(v)), 6, (0, 255, 0), 2)              # detected
            if np.isfinite(pu):
                cv2.circle(out, (int(pu), int(pv)), 2, (0, 0, 255), -1)       # world model reprojected
    cv2.putText(out, f"{cam.id}: world axes (x red, y green, z blue, 10 cm); green = detected, red = model",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", help="calibration recording folder (record_cams.py)")
    ap.add_argument("--make-board", action="store_true", help="write the printable board (PNG + PDF) and exit")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
                    help="--make-board output folder (default packages/data/assets/)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--square-mm", type=float, default=BOARD["square_mm"],
                    help=f"measured square size of YOUR print (default {BOARD['square_mm']})")
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="measured marker size (default: scaled with --square-mm)")
    ap.add_argument("--flat-s", type=float, default=2.0,
                    help="the board lies flat and still during the first FLAT_S seconds (default 2)")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame (default 2)")
    ap.add_argument("--max-dt", type=float, default=0.010, help="max capture time gap of paired frames (s)")
    args = ap.parse_args()

    spec = dict(BOARD)
    if args.square_mm != BOARD["square_mm"]:
        spec["marker_mm"] = args.marker_mm or BOARD["marker_mm"] * args.square_mm / BOARD["square_mm"]
        spec["square_mm"] = args.square_mm
    elif args.marker_mm:
        spec["marker_mm"] = args.marker_mm
    if args.make_board:
        write_board(BOARD, args.out, args.dpi)
        return
    if not args.clip:
        ap.error("pass a calibration clip folder, or --make-board")

    rec = SceneRecording(args.clip)
    cams = rec.cams
    if len(cams) < 1:
        sys.exit(f"{args.clip}: no cameras")
    board = make_board(spec)
    obj_all = board.getChessboardCorners().astype(np.float64)

    print(f"{rec.name}: cameras {list(cams)}, board {spec}")
    dets = {}
    for c, cam in cams.items():
        dets[c] = detect(cam, board, args.stride)
        n = len(range(0, len(cam), args.stride))
        rms = np.median([d[3] for d in dets[c].values()]) if dets[c] else float("nan")
        print(f"  {c}: board in {len(dets[c])}/{n} frames, median PnP reprojection {rms:.2f} px")

    # flat, still frames -> initial T_world_cam per camera
    T_world_cam0 = {}
    flat_by_cam = {}
    for c, cam in cams.items():
        t0 = min(cm.t[0] for cm in cams.values())
        flat = [i for i in dets[c] if cam.t[i] - t0 <= args.flat_s]
        if len(flat) < 3:
            sys.exit(f"{c} saw the board in only {len(flat)} frames of the first {args.flat_s} s: the board must lie "
                     "flat at the start of the clip, visible to every camera (or raise --flat-s)")
        Ts = [T_WORLD_BOARD @ inv(dets[c][i][2]) for i in flat]
        T_world_cam0[c] = mean_pose(Ts)
        spread = np.max([np.linalg.norm(T[:3, 3] - T_world_cam0[c][:3, 3]) for T in Ts]) * 1000
        print(f"  {c}: {len(flat)} flat frames, camera position spread {spread:.1f} mm"
              + ("  WARNING: the board moved during the flat part?" if spread > 10 else ""))
        flat_by_cam[c] = set(flat)

    # observations: frame key = frame index on the reference camera's timeline
    ref = list(cams)[0]
    _, pairing = rec.pair(ref, stride=1, max_dt=args.max_dt)
    obs, flat_keys, T_world_board0, pairs = [], set(), {}, []
    for k, i_ref in enumerate(pairing[ref]):
        idx = {c: int(pairing[c][k]) for c in cams}
        seen = {c: i for c, i in idx.items() if i >= 0 and i in dets[c]}
        if len(cams) > 1 and len(seen) < 2:
            continue                                     # only frames that link the cameras
        if not seen:
            continue
        is_flat = all(i in flat_by_cam[c] for c, i in seen.items())
        if is_flat:
            flat_keys.add(k)
        else:
            c0 = next(iter(seen))
            T_world_board0[k] = T_world_cam0[c0] @ dets[c0][seen[c0]][2]
        for c, i in seen.items():
            obs.append((k, c, dets[c][i][0], dets[c][i][1]))
        if len(seen) >= 2:
            cs = list(cams)[:2]
            if all(c in seen for c in cs):
                pairs.append((seen[cs[0]], seen[cs[1]]))
    if len(cams) == 1:   # single camera: the flat frames alone
        obs = [(i, ref, dets[ref][i][0], dets[ref][i][1]) for i in flat_by_cam[ref]]
        flat_keys = set(flat_by_cam[ref])
    print(f"  joint refinement: {len(obs)} observations, {len(flat_keys)} flat + {len(T_world_board0)} moving frames")
    T_world_cam, _, rms = refine(cams, obs, T_world_cam0, T_world_board0, flat_keys, obj_all)
    for c in cams:
        d = np.linalg.norm(T_world_cam[c][:3, 3] - T_world_cam0[c][:3, 3]) * 1000
        print(f"  {c}: reprojection RMS {rms[c]:.2f} px, moved {d:.1f} mm from the flat-frame estimate, "
              f"position in world {np.round(T_world_cam[c][:3, 3], 3).tolist()} m")

    checks = {}
    if len(cams) >= 2:
        c0, c1 = list(cams)[:2]
        checks["baseline_m"] = round(float(np.linalg.norm(T_world_cam[c0][:3, 3] - T_world_cam[c1][:3, 3])), 4)
        tri = triangulation_check(cams, T_world_cam, pairs, dets, spec)
        checks["triangulation"] = tri
        if tri:
            print(f"  cross-camera check: neighbouring board corners triangulated from {c0}+{c1}: "
                  f"|distance - square| mean {tri['mean_mm']} mm, p95 {tri['p95_mm']} mm over {tri['n']} pairs"
                  + ("  WARNING: > 3 mm, check --square-mm and that the cameras did not move" if tri["mean_mm"] > 3 else ""))
        if len(pairs) < 10:
            print(f"  WARNING: only {len(pairs)} frames where both cameras see the board while it moves: move it "
                  "slowly through the overlap of both views for a better camera<->camera estimate")

    out = {
        "world": "table frame: ChArUco board flat on the table at the start of the clip; origin = top-left "
                 "chessboard corner, x along the rows, y towards the board's top edge, z up; metres",
        "board": spec,
        "clip": rec.name,
        "flat_s": args.flat_s,
        "cameras": {c: {"T_world_cam": T_world_cam[c].tolist(), "K": cams[c].K.tolist(),
                        "dist": cams[c].dist.tolist(), "size": list(cams[c].size), "rms_px": round(rms[c], 3),
                        "n_obs": sum(1 for o in obs if o[1] == c)} for c in cams},
        "T_base_world": None,
        "checks": checks,
    }
    path = os.path.join(rec.dir, "calib.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    imgs = []
    for c, cam in cams.items():
        i = min(flat_by_cam[c])
        imgs.append(draw_check(cam, cam.color(i), T_world_cam[c], dets[c].get(i), obj_all))
    h = min(im.shape[0] for im in imgs)
    cv2.imwrite(os.path.join(rec.dir, "calib_check.jpg"),
                np.hstack([cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]))
    print(f"wrote {path} and calib_check.jpg")


if __name__ == "__main__":
    main()
