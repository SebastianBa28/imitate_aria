"""Settings shared by both pipelines. The first episode of a dataset fixes PROMPTS and N_POINTS for it."""
PROMPTS = ["rubiks cube", "cardboard box"]  # objects to track (Grounding DINO text prompts), in obj_id order
N_POINTS = [250, 450]                # points sampled per object, same order as PROMPTS
LIFT_RADIUS = 7              # px window (2r+1)^2 around a tracked point to read its depth
