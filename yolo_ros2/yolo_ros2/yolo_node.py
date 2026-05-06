import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO

import numpy as np 
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox2D

class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')
        # self.dynamic_classes = [
        #     0, 1, 2, 3, 5, 7, 56 # person, bicycle, car, motorcycle, bus, truck
        # ]
        self.dynamic_classes = [
            0, 1, 2, 3, 5, 7 # person, bicycle, car, motorcycle, bus, truck
        ]
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO("yolov8n-seg.pt")  # or your custom model

        # Subscriber (camera images)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher (results)
        self.publisher = self.create_publisher(
            Image,
            '/yolo/dynamic_mask',
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
        
        self.get_logger().info("YOLO Node Started")

    def image_callback(self, Image):
        # Convert ROS → OpenCV
        
        frame = self.bridge.imgmsg_to_cv2(Image, desired_encoding='bgr8')
        cv2.imshow("RGB", frame)
        cv2.waitKey(1)
        # # Run YOLO
        results = self.model(frame)

        mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        detection_array = Detection2DArray()
        detection_array.header = Image.header

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for seg, cls, box, conf in zip(masks, classes, boxes, confs):

                if int(cls) in self.dynamic_classes:

                    x1, y1, x2, y2 = box.astype(int)

                    # --- Resize segmentation ---
                    seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))

                    # --- Dynamic mask ---
                    mask[seg > 0.5] = 0

                    # =========================
                    # Detection message
                    # =========================
                    det = Detection2D()

                    det.header = Image.header

                    # --- Bounding box ---
                    bbox = BoundingBox2D()

                    bbox.center.position.x = float((x1 + x2) / 2.0)
                    bbox.center.position.y = float((y1 + y2) / 2.0)

                    bbox.size_x = float(x2 - x1)
                    bbox.size_y = float(y2 - y1)

                    det.bbox = bbox

                    # --- Class + confidence ---
                    hypothesis = ObjectHypothesisWithPose()

                    hypothesis.hypothesis.class_id = str(int(cls))
                    hypothesis.hypothesis.score = float(conf)

                    det.results.append(hypothesis)

                    detection_array.detections.append(det)

                    # =========================
                    # Visualization
                    # =========================
                    cv2.rectangle(frame,
                                (x1,y1),
                                (x2,y2),
                                (0,255,0),
                                2)

                    label = f"{int(cls)} {conf:.2f}"

                    cv2.putText(frame,
                                label,
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,0),
                                2)


        vis = mask*255
        cv2.imshow("mask", vis)
        cv2.waitKey(1)

        cv2.imshow("bbox", frame)
        cv2.waitKey(1)

        dynamic_mask = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.publisher.publish(dynamic_mask)
        self.detection_pub.publish(detection_array)
        
        # self.publisher.publish("Publishing Detections")
        # self.get_logger().info("Publishing: ")

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()