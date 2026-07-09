import sys, time, json
import cv2
import numpy as np
from ultralytics import YOLO

FRAME_PATH = "/root/colcon_ws/src/orb_slam3_ros2_wrapper/datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846315.298000.png"
OUT_DIR = "/home/orb/ORB_SLAM3/cpp_yolo_prototype"

dynamic_classes = [0, 1, 2, 3, 5, 7]

model = YOLO("/home/orb/ORB_SLAM3/models/yolov8n-seg.pt")

frame = cv2.imread(FRAME_PATH)
assert frame is not None, f"failed to read {FRAME_PATH}"

# Warm up (first call includes lazy init overhead)
_ = model(frame, verbose=False)

t0 = time.time()
results = model(frame, verbose=False)
t1 = time.time()
print(f"inference took {(t1-t0)*1000:.1f} ms")

mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
dets = []

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    for seg, cls, box, conf in zip(masks, classes, boxes, confs):
        if int(cls) in dynamic_classes:
            x1, y1, x2, y2 = box.astype(int)
            seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))
            mask[seg > 0.5] = 0
            dets.append({
                "class_id": int(cls),
                "conf": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

cv2.imwrite(f"{OUT_DIR}/ref_mask.png", mask * 255)
with open(f"{OUT_DIR}/ref_detections.json", "w") as f:
    json.dump(dets, f, indent=2)

print(f"detections: {len(dets)}")
for d in dets:
    print(d)
print(f"dynamic pixel count: {(mask==0).sum()} / {mask.size}")
