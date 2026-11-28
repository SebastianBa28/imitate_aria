"""Step 1-2 of AINA: Grounding DINO box per prompt, then a SAM 2 mask per box, on ONE frame."""
import cv2
import numpy as np
import torch

from aina.overlay import OVERLAY_COLORS


def boxes_grounding_dino(rgb, prompts, device, box_thr):
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    img = Image.fromarray(rgb)
    boxes = []
    for p in prompts:
        text = p.lower().strip().rstrip(".") + "."
        inp = proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        try:
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=box_thr, target_sizes=[img.size[::-1]])[0]
        except TypeError:  # older transformers
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_thr, target_sizes=[img.size[::-1]])[0]
        if len(res["boxes"]) == 0:
            print(f"  Grounding DINO: nothing for '{p}' (lower --box-thr or pick another start frame with --start)")
            continue
        k = int(res["scores"].argmax())
        b = res["boxes"][k].cpu().numpy().tolist()
        print(f"  '{p}': box {[round(v) for v in b]} score {float(res['scores'][k]):.2f}")
        boxes.append((p, b))
    del model
    torch.cuda.empty_cache()
    return boxes


# ----------------------------------------------------------------------------- step 2: segmentation-frame masks
def masks_sam2(rgb, boxes):
    """SAM 2 in image mode on one frame only: one (H,W) bool mask per box, same order."""
    from ultralytics import SAM

    h, w = rgb.shape[:2]
    res = SAM("sam2.1_l.pt")(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bboxes=[b for _, b in boxes], verbose=False)[0]
    if res.masks is None:
        return [np.zeros((h, w), bool) for _ in boxes]
    masks = []
    for m in res.masks.data.cpu().numpy():
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m.astype(bool))
    for (p, _), m in zip(boxes, masks):
        print(f"  '{p}': mask {m.sum()} px ({m.mean():.3%} of frame)")
    return masks


def draw_segmentation(rgb, boxes, masks0):
    """Segmentation frame with the Grounding DINO boxes and SAM 2 masks drawn on (BGR, frame resolution)."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k, ((name, b), m) in enumerate(zip(boxes, masks0)):
        c = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
        out[m] = (0.45 * out[m] + 0.55 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, c, 2)
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{name} ({m.sum()} px)"
        cv2.putText(out, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    return out
