from ultralytics import YOLO
import cv2
import numpy as np
import sys

model = YOLO("yolov8n-seg.pt")

# dynamic classes in COCO
dynamic_classes = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

# image_path = sys.argv[1]
image_path = "/root/colcon_ws/src/orb_slam3_ros2_wrapper/datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846342.447907.png"

img = cv2.imread(image_path)
results = model(img)

print(results[0].masks.data.shape)
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

if results[0].masks is not None:

    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    print("mask tensor shape:", masks.shape)
    print("classes:", classes)
    for seg, cls in zip(masks, classes):
        
        if int(cls) in dynamic_classes:

            seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
            mask[seg > 0.5] = 0

# mask_vis = (mask * 255).astype(np.uint8)
# cv2.imshow("seg resized", mask_vis)
# cv2.waitKey(1)

cv2.imwrite("/home/orb/ORB_SLAM3/tmp/dynamic_mask.png", mask)