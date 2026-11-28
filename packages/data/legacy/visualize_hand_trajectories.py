#!/usr/bin/env python3
"""
Interactive 3D viewer for hand and object trajectories in the Table frame.

Shows:
- AprilTag table origin (0,0,0) and coordinate triad
- glasses path + current pose triad
- hand skeleton, fingertips, wrist pose triad, and recent motion trails
- 3D cube wireframe + center point (green)
- 3D tray polygon + static session boundary (magenta)

Usage:
    python visualize_hand_trajectories.py recording_hands.npz
    python visualize_hand_trajectories.py recording_hands.npz --objects recording_objects.npz
"""
import argparse
from pathlib import Path

import numpy as np

# Landmark connectivity for drawing the hand
BONES = [
    (5, 6), (6, 7), (7, 0),                 # thumb
    (5, 8), (8, 9), (9, 10), (10, 1),       # index
    (5, 11), (11, 12), (12, 13), (13, 2),   # middle
    (5, 14), (14, 15), (15, 16), (16, 3),   # ring
    (5, 17), (17, 18), (18, 19), (19, 4),   # pinky
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
HAND_COLORS = {"left": "tab:blue", "right": "tab:red"}
SIDES = ("right",)  # right hand only; use ("left", "right") to show both

# Cube 12 edges for 8 vertices ordered by binary coordinates (x, y, z)
CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def insert_breaks(arr, breaks):
    """Insert NaN rows at the given indices so lines are not drawn across them."""
    if len(breaks) == 0:
        return arr
    return np.insert(arr, breaks, np.nan, axis=0)


def skeleton_xyz(lm):
    """(21,3) landmarks -> (3*len(BONES),3) polyline with NaN separators."""
    pts = []
    for a, b in BONES:
        pts.extend([lm[a], lm[b], [np.nan] * 3])
    return np.array(pts)


def cube_wireframe_xyz(verts):
    """(8,3) cube vertices -> (3*12, 3) polyline with NaN separators."""
    if not np.isfinite(verts).all():
        return np.full((1, 3), np.nan)
    pts = []
    for a, b in CUBE_EDGES:
        pts.extend([verts[a], verts[b], [np.nan] * 3])
    return np.array(pts)


def tray_poly_xyz(corners):
    """(4,3) tray corners -> (5,3) closed polyline loop."""
    if not np.isfinite(corners).all():
        return np.full((1, 3), np.nan)
    return np.vstack([corners, corners[0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hands_npz", help="file written by extract_hand_trajectories.py")
    ap.add_argument("--objects", help="file written by extract_object_trajectories.py (auto-detected if omitted)")
    ap.add_argument("--trail", type=float, default=2.0, help="length of the bright trail in seconds (default 2)")
    ap.add_argument("--step", type=int, default=1, help="use every Nth sample for the slider (default 1)")
    ap.add_argument("--fps", type=float, default=20.0, help="playback speed in slider steps per second")
    ap.add_argument("--png", help="save a static overview image (full trails) and exit, no window")
    ap.add_argument("--no-skeleton", action="store_true", help="hide the hand skeleton, show fingertips only")
    args = ap.parse_args()

    import matplotlib
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    # --- 1. Load Hands Data ---
    d_hands = np.load(args.hands_npz, allow_pickle=False)
    t = d_hands["time_s"]
    N = len(t)
    sess = d_hands["session_idx"]
    breaks = np.where(np.diff(sess) != 0)[0] + 1

    sides = SIDES
    glasses = d_hands["glasses_pos"]
    lm_w = {s: d_hands[f"{s}_landmarks_world"] for s in sides}
    wrist_p = {s: d_hands[f"{s}_wrist_pos_world"] for s in sides}
    wrist_T = {s: d_hands[f"{s}_T_world_wrist"] for s in sides}
    tips = {s: d_hands[f"{s}_fingertips_world"] for s in sides}
    T_gl = d_hands["T_world_device"]

    # --- 2. Load Objects Data (If Available) ---
    obj_path = args.objects
    if not obj_path:
        candidate = Path(args.hands_npz).with_name(
            Path(args.hands_npz).stem.replace("_hands", "") + "_objects.npz"
        )
        if candidate.exists():
            obj_path = str(candidate)

    cube_valid = np.zeros(N, dtype=bool)
    cube_center = np.full((N, 3), np.nan)
    cube_verts = np.full((N, 8, 3), np.nan)

    tray_valid = np.zeros(N, dtype=bool)
    tray_corners = np.full((N, 4, 3), np.nan)
    tray_static = None

    if obj_path and Path(obj_path).exists():
        print(f"Loading object data from {obj_path}")
        d_obj = np.load(obj_path, allow_pickle=False)
        s_idx = d_obj["sample_idx"].astype(int)

        cube_valid[s_idx] = d_obj["cube_valid"]
        cube_center[s_idx] = d_obj["cube_center_world"]
        cube_verts[s_idx] = d_obj["cube_vertices_world"]

        tray_valid[s_idx] = d_obj["tray_valid"]
        tray_corners[s_idx] = d_obj["tray_corners_world"]
        if "tray_corners_static_world" in d_obj:
            tray_static = d_obj["tray_corners_static_world"]
    else:
        print("No object tracking NPZ file found or provided. Displaying hands only.")

    # --- 3. Load & Apply Table Frame Transformation (MOVED UP HERE!) ---
    table_path = Path(args.hands_npz).with_name(Path(args.hands_npz).stem.replace("_hands", "") + "_table.npz")
    T_table_world = None

    if table_path.exists():
        print(f"Loading static table frame from {table_path}")
        d_tbl = np.load(table_path)
        T_table_world = d_tbl["T_table_world"]

    def transform_pts(pts):
        """Transform (N, 3) 3D points from VIO World -> Table Frame"""
        if T_table_world is None or not np.isfinite(pts).any():
            return pts
        shape = pts.shape
        pts_flat = pts.reshape(-1, 3)
        pts_h = np.hstack([pts_flat, np.ones((len(pts_flat), 1))])
        pts_tbl = (T_table_world @ pts_h.T).T[:, :3]
        return pts_tbl.reshape(shape)

    def transform_pose(T_matrix):
        """Transform 4x4 SE(3) pose matrices (N, 4, 4) from VIO World -> Table Frame"""
        if T_table_world is None or not np.isfinite(T_matrix).any():
            return T_matrix
        if T_matrix.ndim == 2:
            return T_table_world @ T_matrix
        elif T_matrix.ndim == 3:
            return np.matmul(T_table_world, T_matrix)
        return T_matrix

    # Apply 3D point transformations
    glasses = transform_pts(glasses)
    for s in sides:
        lm_w[s] = transform_pts(lm_w[s])
        wrist_p[s] = transform_pts(wrist_p[s])
        tips[s] = transform_pts(tips[s])

    cube_center = transform_pts(cube_center)
    cube_verts = transform_pts(cube_verts)
    tray_corners = transform_pts(tray_corners)
    if tray_static is not None:
        tray_static = transform_pts(tray_static)

    # Apply 4x4 SE(3) matrix transformations (FIXES THE FLOATING AXES!)
    T_gl = transform_pose(T_gl)
    for s in sides:
        wrist_T[s] = transform_pose(wrist_T[s])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.14)

    # --- 4. Axis limits / Aspect ratio (Calculated on transformed data) ---
    allpts = [glasses] + [lm_w[s].reshape(-1, 3) for s in sides]
    if cube_valid.any():
        allpts.append(cube_center[cube_valid])
    if tray_valid.any():
        allpts.append(tray_corners[tray_valid].reshape(-1, 3))

    if T_table_world is not None:
        allpts.append(np.array([[0.0, 0.0, 0.0]]))
    allpts = np.vstack(allpts)
    allpts = allpts[np.isfinite(allpts).all(axis=1)]

    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 0.2)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

    # --- 5. Static Faint Full Paths ---
    def path(arr, **kw):
        a = insert_breaks(arr, breaks)
        return ax.plot(a[:, 0], a[:, 1], a[:, 2], **kw)[0]

    path(glasses, color="0.4", lw=1.5, alpha=0.5, label="glasses")
    for s in sides:
        path(wrist_p[s], color=HAND_COLORS[s], lw=1.2, alpha=0.25)
        for k in range(5):
            path(tips[s][:, k, :], color=FINGER_COLORS[k], lw=0.8, alpha=0.25)

    if cube_valid.any():
        path(cube_center, color="limegreen", lw=1.0, alpha=0.35, label="cube path")

    # Static Tray bounds per session
    if tray_static is not None:
        for s_idx in range(len(tray_static)):
            tc = tray_static[s_idx]
            if np.isfinite(tc).all():
                t_poly = tray_poly_xyz(tc)
                ax.plot(t_poly[:, 0], t_poly[:, 1], t_poly[:, 2], "--", color="magenta", lw=1.2, alpha=0.4,
                        label="tray static" if s_idx == 0 else None)

    # Draw Table Tag Origin (0,0,0) and 10cm Triad
    if T_table_world is not None:
        ax.plot([0], [0], [0], "k*", ms=12, label="Table Tag (0,0,0)")
        ax.plot([0, 0.1], [0, 0], [0, 0], "r-", lw=3)
        ax.plot([0, 0], [0, 0.1], [0, 0], "g-", lw=3)
        ax.plot([0, 0], [0, 0], [0, 0.1], "b-", lw=3)

    # --- 6. Dynamic Artists ---
    (glasses_dot,) = ax.plot([], [], [], "o", color="black", ms=7, label="glasses (now)")
    glasses_axes = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in ("r", "g", "b")]

    hand_art = {}
    for s in sides:
        art = {}
        art["skel"] = ax.plot([], [], [], "-", color=HAND_COLORS[s], lw=2, label=f"{s} hand")[0]
        art["wrist"] = ax.plot([], [], [], "o", color=HAND_COLORS[s], ms=8)[0]
        art["tipdots"] = [ax.plot([], [], [], "o", color=FINGER_COLORS[k], ms=5,
                                  label=FINGER_NAMES[k] if s == SIDES[-1] else None)[0] for k in range(5)]
        art["axes"] = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in ("r", "g", "b")]
        art["wtrail"] = ax.plot([], [], [], "-", color=HAND_COLORS[s], lw=3)[0]
        art["ttrail"] = [ax.plot([], [], [], "-", color=FINGER_COLORS[k], lw=2)[0] for k in range(5)]
        hand_art[s] = art

    # Dynamic Object Artists
    cube_wire = ax.plot([], [], [], "-", color="limegreen", lw=2, label="cube")[0]
    cube_dot = ax.plot([], [], [], "o", color="limegreen", ms=5)[0]
    tray_poly_art = ax.plot([], [], [], "-", color="magenta", lw=2.5, label="tray (live)")[0]

    title = ax.set_title("")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    def set3d(line, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

    def draw_triad(lines, T, length):
        if not np.isfinite(T).all():
            for ln in lines:
                set3d(ln, np.full((2, 3), np.nan))
            return
        o = T[:3, 3]
        for i, ln in enumerate(lines):
            set3d(ln, [o, o + length * T[:3, i]])

    trail_s = np.inf if args.png else args.trail

    def update(k):
        k = int(k)
        j0 = int(np.searchsorted(t, t[k] - trail_s)) if np.isfinite(trail_s) else 0

        # Glasses & Hands
        set3d(glasses_dot, glasses[k])
        draw_triad(glasses_axes, T_gl[k], 0.10)
        status = []
        for s in sides:
            art = hand_art[s]
            lm = lm_w[s][k]
            ok = np.isfinite(lm).all()
            status.append(f"{s} hand: {'tracked' if ok else 'lost'}")
            if args.no_skeleton or not ok:
                set3d(art["skel"], np.full((1, 3), np.nan))
            else:
                set3d(art["skel"], skeleton_xyz(lm))
            set3d(art["wrist"], wrist_p[s][k])
            for f in range(5):
                set3d(art["tipdots"][f], tips[s][k, f])
                set3d(art["ttrail"][f], tips[s][j0:k + 1, f, :])
            set3d(art["wtrail"], wrist_p[s][j0:k + 1])
            draw_triad(art["axes"], wrist_T[s][k], 0.05)

        # Objects
        c_ok = cube_valid[k]
        status.append(f"cube: {'tracked' if c_ok else 'lost'}")
        if c_ok:
            set3d(cube_wire, cube_wireframe_xyz(cube_verts[k]))
            set3d(cube_dot, cube_center[k])
        else:
            set3d(cube_wire, np.full((1, 3), np.nan))
            set3d(cube_dot, np.full((1, 3), np.nan))

        tr_ok = tray_valid[k]
        if tr_ok:
            set3d(tray_poly_art, tray_poly_xyz(tray_corners[k]))
        else:
            set3d(tray_poly_art, np.full((1, 3), np.nan))

        title.set_text(f"t = {t[k]:.2f} s   (sample {k + 1}/{N}, VIO session {int(sess[k])})   "
                       + "   ".join(status))
        fig.canvas.draw_idle()

    if args.png:
        update(N - 1)
        title.set_text(f"Full recording, {t[-1]:.1f} s")
        fig.savefig(args.png, dpi=150)
        print(f"Saved {args.png}")
        return

    idxs = np.arange(0, N, max(1, args.step))
    ax_sl = fig.add_axes([0.15, 0.04, 0.70, 0.03])
    slider = Slider(ax_sl, "time", 0, len(idxs) - 1, valinit=0, valstep=1)
    slider.valtext.set_text("0.00 s")

    def on_slider(v):
        k = idxs[int(v)]
        slider.valtext.set_text(f"{t[k]:.2f} s")
        update(k)

    slider.on_changed(on_slider)

    ax_btn = fig.add_axes([0.02, 0.035, 0.08, 0.04])
    button = Button(ax_btn, "Play")
    timer = fig.canvas.new_timer(interval=int(1000 / args.fps))
    state = {"playing": False}

    def advance():
        v = int(slider.val) + 1
        if v >= len(idxs):
            v = 0
        slider.set_val(v)

    timer.add_callback(advance)

    def on_button(_):
        state["playing"] = not state["playing"]
        if state["playing"]:
            timer.start()
            button.label.set_text("Pause")
        else:
            timer.stop()
            button.label.set_text("Play")
        fig.canvas.draw_idle()

    button.on_clicked(on_button)

    update(idxs[0])
    plt.show()


if __name__ == "__main__":
    main()