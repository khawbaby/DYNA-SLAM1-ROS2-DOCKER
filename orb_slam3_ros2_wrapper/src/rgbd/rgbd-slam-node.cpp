/**
 * @file rgbd-slam-node.cpp
 * @brief Implementation of the RgbdSlamNode Wrapper class.
 * @author Suchetan R S (rssuchetan@gmail.com)
 */
#include "rgbd-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <deque>
namespace ORB_SLAM3_Wrapper
{
    using namespace WrapperTypeConversions;
    RgbdSlamNode::RgbdSlamNode(const std::string &strVocFile,
                               const std::string &strSettingsFile,
                               ORB_SLAM3::System::eSensor sensor)
        : SlamNodeBase("ORB_SLAM3_RGBD_ROS2", strVocFile, strSettingsFile, sensor)
    {
        // Declare parameters (topic names)
        this->declare_parameter("rgb_image_topic_name", rclcpp::ParameterValue("/camera/camera/color/image_raw"));
        this->declare_parameter("depth_image_topic_name", rclcpp::ParameterValue("/camera/camera/depth/image_rect_raw"));
        this->declare_parameter("yolo_model_path", rclcpp::ParameterValue("/home/orb/ORB_SLAM3/models/yolov8n-seg.onnx"));
        this->declare_parameter("yolo_use_cuda", rclcpp::ParameterValue(true));

        // Synced ROS Subscribers
        rgbSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("rgb_image_topic_name").as_string());
        depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("depth_image_topic_name").as_string());

        // In-process YOLOv8-seg detector, run synchronously per-frame inside
        // RGBDCallback below instead of via async mask/detections topics -
        // eliminates the timing gap between when a detection was computed
        // and which RGB frame it actually gets applied to.
        yoloDetector_ = std::make_unique<YoloSegDetector>(
            this->get_parameter("yolo_model_path").as_string(),
            this->get_parameter("yolo_use_cuda").as_bool());

        syncApproximate_ = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(approximate_sync_policy(10), *rgbSub_, *depthSub_);
        syncApproximate_->registerCallback(&RgbdSlamNode::RGBDCallback, this);

        RCLCPP_INFO(this->get_logger(), "CONSTRUCTOR END!");
    }

    RgbdSlamNode::~RgbdSlamNode()
    {
        rgbSub_.reset();
        depthSub_.reset();

        RCLCPP_INFO(this->get_logger(), "DESTRUCTOR!");
    }

    void RgbdSlamNode::RGBDCallback(const sensor_msgs::msg::Image::SharedPtr msgRGB, const sensor_msgs::msg::Image::SharedPtr msgD)
    {
        cv_bridge::CvImageConstPtr cvRGB;
        cv_bridge::CvImageConstPtr cvD;

        // =========================================
        // RGB IMAGE
        // =========================================
        try
        {
            cvRGB = cv_bridge::toCvShare(msgRGB);
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception RGB!" << std::endl;
            return;
        }

        // =========================================
        // DEPTH IMAGE
        // =========================================
        try
        {
            cvD = cv_bridge::toCvShare(msgD);
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception D!" << std::endl;
            return;
        }

        // =========================================
        // YOLO SEGMENTATION (synchronous, this exact frame)
        // =========================================
        YoloSegResult yoloResult = yoloDetector_->Run(cvRGB->image);

        // =========================================
        // TRACK
        // =========================================
        auto start = std::chrono::high_resolution_clock::now();
        Sophus::SE3f Tcw;
        try {
            Tcw = interface()->slam()->TrackRGBD(
                cvRGB->image,
                cvD->image,
                yoloResult.mask,
                yoloResult.detections,
                stampToSec(msgRGB->header.stamp)
            );
        } catch (const std::exception& e) {
            RCLCPP_ERROR(
                this->get_logger(),
                "ORB-SLAM3 exception: %s",
                e.what()
            );
            return;
        } catch(...) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Unknown ORB-SLAM3 fatal exception"
            );
            return;
        }
        // =========================================
        // FPS
        // =========================================
        {
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms =
                std::chrono::duration<double, std::milli>(
                    end - start).count();

            times.push_back(time_ms);

            if(times.size() > window)
                times.pop_front();

            double sum = 0;

            for(double t : times)
                sum += t;

            double avg = sum / times.size();

            double fps = 1000.0 / avg;

            // std::cout << "FPS (smoothed): "
            //         << fps
            //         << std::endl;
        }

        // =========================================
        // PUBLISH TRACKED POSE
        // =========================================
        if (interface()->processTrackedPose(Tcw))
        {
            this->onTracked(msgRGB->header);
        }
    }
}
