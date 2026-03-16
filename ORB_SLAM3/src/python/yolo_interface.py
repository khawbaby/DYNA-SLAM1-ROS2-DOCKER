from ultralytics import YOLO
import numpy as np
import cv2

class YoloSegment:

    def __init__(self):
        self.model = YOLO("yolov8n-seg.pt")

        self.dynamic_classes = [
            0, 1, 2, 3, 5, 7  # person, bicycle, car, motorcycle, bus, truck
        ]

    def segment(self, image):

        results = self.model(image)

        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if results[0].masks is not None:

            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for seg, cls in zip(masks, classes):

                if int(cls) in self.dynamic_classes:
                    seg = cv2.resize(seg, (image.shape[1], image.shape[0]))
                    mask[seg > 0.5] = 0

        return mask