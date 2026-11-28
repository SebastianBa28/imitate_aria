# Batch health: nov27 recordings

## Summary
- 27 recordings, 1370 frames, 137 s at 10 fps; one VIO session each
- frames: median 49 (min 40, max 97)
- start: median 8 (min 6, max 24)
- pose_pct: median 97.6 (min 92, max 100)
- right_pct: median 97.6 (min 85.7, max 100)
- right_longest_gap: median 1 (min 0, max 2)
- right_tip_speed_max: median 1.37 (min 0.76, max 1.93)
- cube_vis_pct: median 87.5 (min 67.6, max 100)
- cube_3d_given_vis_pct: median 100 (min 73.5, max 100)
- cube_strays: median 5 (min 0, max 74)
- box_vis_pct: median 90.7 (min 62.4, max 95.9)
- box_3d_given_vis_pct: median 98.6 (min 77.5, max 100)
- box_strays: median 4 (min 0, max 71)

## Flagged (thresholds: right hand <90%, gap >=3 fr, depth <90% of visible points, strays >5% of points, cube visible <75%)
- **rubiks-drop_nov27**: cube depth only 81.5% of visible; box depth only 85.3% of visible
- **rubiks_nov27_2**: cube strays 17/250
- **rubiks_nov27_3**: cube strays 19/250
- **rubiks_nov27_5**: cube visible 67.6%
- **rubiks_nov27_7**: right hand tracked 85.7%
- **rubiks_nov27_8**: cube depth only 73.5% of visible; box depth only 77.5% of visible
- **rubiks_nov27_9**: cube depth only 75.5% of visible; box depth only 82.4% of visible; cube strays 27/250
- **rubiks_nov27_13**: cube strays 36/250
- **rubiks_nov27_14**: cube strays 74/250; box strays 71/450
- **rubiks_nov27_15**: cube strays 15/250
- **rubiks_nov27_19**: cube visible 71.4%
- **rubiks_nov27_22**: cube strays 20/250

## Per recording
| rec | frames | start | no_pose_frames | right_pct | right_longest_gap | right_jumps_gt2ms | left_pct | cube_vis_pct | cube_3d_given_vis_pct | cube_strays | box_vis_pct | box_3d_given_vis_pct | box_strays | cube_depth_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rubiks-drop_nov27 | 70 | 8 | [1, 57] | 97.1 | 1 | 0 | 0.0 | 93.4 | 81.5 | 5 | 95.9 | 85.3 | 5 | 0.48 |
| rubiks-drop_nov27_1 | 97 | 9 | [1, 96] | 97.9 | 1 | 0 | 0.0 | 83.5 | 100.0 | None | 66.5 | 98.6 | None | 0.55 |
| rubiks_nov27 | 48 | 11 | [1, 2] | 95.8 | 2 | 0 | 22.9 | 85.1 | 100.0 | 11 | 94.2 | 100.0 | 4 | 0.67 |
| rubiks_nov27_1 | 59 | 10 | [1, 2] | 96.6 | 2 | 0 | 96.6 | 100.0 | 100.0 | 1 | 93.6 | 100.0 | 2 | 0.69 |
| rubiks_nov27_2 | 40 | 8 | [1] | 97.5 | 1 | 0 | 35.0 | 96.2 | 100.0 | 17 | 93.3 | 100.0 | 2 | 0.67 |
| rubiks_nov27_3 | 43 | 24 | [] | 100.0 | 0 | 0 | 0.0 | 97.5 | 100.0 | 19 | 91.7 | 100.0 | 9 | 0.65 |
| rubiks_nov27_4 | 40 | 8 | [1] | 95.0 | 2 | 0 | 0.0 | 86.6 | 100.0 | 5 | 88.3 | 98.1 | 3 | 0.59 |
| rubiks_nov27_5 | 52 | 8 | [1, 26, 50] | 94.2 | 1 | 0 | 0.0 | 67.6 | 98.9 | 4 | 90.8 | 94.7 | 7 | 0.55 |
| rubiks_nov27_6 | 50 | 10 | [1, 22] | 96.0 | 1 | 0 | 0.0 | 78.2 | 100.0 | 2 | 90.7 | 99.6 | 4 | 0.61 |
| rubiks_nov27_7 | 49 | 8 | [1, 48] | 85.7 | 0 | 0 | 0.0 | 85.7 | 100.0 | None | 91.7 | 99.0 | None | 0.55 |
| rubiks_nov27_8 | 41 | 8 | [1, 24] | 95.1 | 1 | 0 | 0.0 | 87.5 | 73.5 | 0 | 94.6 | 77.5 | 1 | 0.52 |
| rubiks_nov27_9 | 42 | 7 | [1] | 97.6 | 1 | 0 | 0.0 | 87.7 | 75.5 | 27 | 89.5 | 82.4 | 0 | 0.49 |
| rubiks_nov27_10 | 49 | 10 | [1] | 98.0 | 1 | 0 | 0.0 | 88.9 | 100.0 | 2 | 90.7 | 100.0 | 10 | 0.59 |
| rubiks_nov27_11 | 54 | 9 | [1] | 98.1 | 1 | 0 | 0.0 | 80.7 | 99.4 | None | 78.2 | 98.4 | None | 0.6 |
| rubiks_nov27_12 | 50 | 9 | [1, 2, 15, 49] | 92.0 | 2 | 0 | 0.0 | 85.2 | 91.5 | None | 93.4 | 90.1 | None | 0.52 |
| rubiks_nov27_13 | 54 | 8 | [1] | 98.1 | 1 | 0 | 98.1 | 94.8 | 100.0 | 36 | 94.0 | 99.2 | 3 | 0.62 |
| rubiks_nov27_14 | 53 | 9 | [1, 28, 39] | 94.3 | 1 | 0 | 49.1 | 92.7 | 100.0 | 74 | 81.2 | 100.0 | 71 | 0.58 |
| rubiks_nov27_15 | 52 | 6 | [1] | 98.1 | 1 | 0 | 98.1 | 83.9 | 100.0 | 15 | 89.0 | 99.8 | 12 | 0.57 |
| rubiks_nov27_16 | 59 | 8 | [1, 58] | 96.6 | 1 | 0 | 78.0 | 93.4 | 100.0 | None | 87.9 | 100.0 | None | 0.6 |
| rubiks_nov27_17 | 48 | 8 | [1] | 97.9 | 1 | 0 | 97.9 | 85.3 | 100.0 | 5 | 94.9 | 99.3 | 8 | 0.55 |
| rubiks_nov27_18 | 41 | 8 | [1] | 97.6 | 1 | 0 | 97.6 | 88.4 | 100.0 | 8 | 62.4 | 98.3 | 3 | 0.57 |
| rubiks_nov27_19 | 54 | 7 | [1] | 98.1 | 1 | 0 | 98.1 | 71.4 | 100.0 | 5 | 70.4 | 98.6 | 17 | 0.54 |
| rubiks_nov27_20 | 40 | 7 | [1] | 97.5 | 1 | 0 | 97.5 | 84.1 | 100.0 | 4 | 66.7 | 90.3 | 2 | 0.54 |
| rubiks_nov27_21 | 42 | 8 | [1] | 97.6 | 1 | 0 | 92.9 | 92.4 | 99.9 | 1 | 72.0 | 91.1 | 10 | 0.47 |
| rubiks_nov27_22 | 47 | 8 | [1] | 97.9 | 1 | 0 | 97.9 | 89.1 | 100.0 | 20 | 93.1 | 96.5 | 0 | 0.48 |
| rubiks_nov27_23 | 43 | 7 | [1] | 97.7 | 1 | 0 | 97.7 | 82.0 | 100.0 | 5 | 76.9 | 100.0 | 1 | 0.49 |
| rubiks_nov27_24 | 53 | 9 | [1] | 98.1 | 1 | 0 | 98.1 | 91.4 | 99.9 | 2 | 68.9 | 97.6 | 11 | 0.43 |

_pct = % of point-frames. 3d_given_vis = visible points (in frames with a pose) that got a 3D position. strays = points whose distance to the object's median changed >5 cm between first and last frame (rigid-object check). no_pose_frames = frame indices with no VIO pose (NaN)._
