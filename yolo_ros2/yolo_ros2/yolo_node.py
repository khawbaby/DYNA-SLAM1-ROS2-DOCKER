import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO

import numpy as np 

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

        self.get_logger().info("YOLO Node Started")

    def image_callback(self, Image):
        # Convert ROS → OpenCV
        
        frame = self.bridge.imgmsg_to_cv2(Image, desired_encoding='bgr8')
        cv2.imshow("RGB", frame)
        cv2.waitKey(1)
        # # Run YOLO
        results = self.model(frame)

        mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if results[0].masks is not None:

            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for seg, cls in zip(masks, classes):

                if int(cls) in self.dynamic_classes:
                    seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))
                    mask[seg > 0.5] = 0


        vis = mask*255
        cv2.imshow("mask", vis)
        cv2.waitKey(1)
        dynamic_mask = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.publisher.publish(dynamic_mask)
    
        # self.publisher.publish("Publishing Detections")
        # self.get_logger().info("Publishing: ")

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()