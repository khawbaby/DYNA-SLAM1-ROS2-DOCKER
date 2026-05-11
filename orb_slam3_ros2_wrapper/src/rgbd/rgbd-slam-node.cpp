/**
 * @file rgbd-slam-node.cpp
 * @brief Implementation of the RgbdSlamNode Wrapper class.
 * @author Suchetan R S (rssuchetan@gmail.com)
 */
#include "rgbd-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
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
        this->declare_parameter("dynamic_mask_topic_name", rclcpp::ParameterValue("yolo/dynamic_mask"));
        this->declare_parameter("detections_topic_name", rclcpp::ParameterValue("yolo/detections"));

        // Synced ROS Subscribers
        rgbSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("rgb_image_topic_name").as_string());
        depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("depth_image_topic_name").as_string());
        
        // Async Mask Subscriber
        maskSub_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("dynamic_mask_topic_name").as_string(),
            10,
            std::bind(&RgbdSlamNode::MaskCallback, this, std::placeholders::_1)
        );

        detectionsSub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            this->get_parameter("detections_topic_name").as_string(),
            10,
            std::bind(&RgbdSlamNode::DetectionsCallback,
                    this,
                    std::placeholders::_1)
        );

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
        // WAIT FOR FIRST MASK
        // =========================================
        if (!mask_received_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for first mask..."
            );
            return;
        }

        // =========================================
        // WAIT FOR FIRST DETECTIONS
        // =========================================
        if (!detections_received_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for first detections..."
            );
            return;
        }

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
        // GET LATEST MASK (THREAD SAFE)
        // =========================================
        cv::Mat mask_copy;

        {
            std::lock_guard<std::mutex> lock(mask_mutex_);
            mask_copy = latest_mask_.clone();
        }

        // =========================================
        // GET LATEST DETECTIONS (THREAD SAFE)
        // =========================================
        vision_msgs::msg::Detection2DArray detections_copy;

        {
            std::lock_guard<std::mutex> lock(detections_mutex_);
            detections_copy = latest_detections_;
        }

        // =========================================
        // HANDLE EMPTY MASK
        // =========================================
        if (mask_copy.empty()) {
            mask_copy = cv::Mat::ones(
                cvRGB->image.size(),
                CV_8UC1
            );
        }

        // =========================================
        // OPTIONAL DEBUG PRINT
        // =========================================
        // std::cout << "Detections: "
        //           << detections_copy.detections.size()
        //           << std::endl;

        // =========================================
        // TRACK
        // =========================================
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<ORB_SLAM3::Detection> slamDetections;
        for(const auto& det : detections_copy.detections)
        {
            ORB_SLAM3::Detection d;

            float cx = det.bbox.center.position.x;
            float cy = det.bbox.center.position.y;

            float w = det.bbox.size_x;
            float h = det.bbox.size_y;

            int x1 = static_cast<int>(cx - w * 0.5f);
            int y1 = static_cast<int>(cy - h * 0.5f);

            d.bbox = cv::Rect(
                x1,
                y1,
                static_cast<int>(w),
                static_cast<int>(h)
            );

            if(!det.results.empty())
            {
                d.class_id =
                    std::stoi(det.results[0].hypothesis.class_id);

                d.confidence =
                    det.results[0].hypothesis.score;
            }

            slamDetections.push_back(d);
        }
        auto Tcw = interface()->slam()->TrackRGBD(
            cvRGB->image,
            cvD->image,
            mask_copy,
            slamDetections,
            stampToSec(msgRGB->header.stamp)
        );

        // =========================================
        // FPS
        // =========================================
        if (!mask_copy.empty())
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

    void RgbdSlamNode::MaskCallback(const sensor_msgs::msg::Image::SharedPtr msgMask)
    {
        try
        {
            auto cvMask = cv_bridge::toCvShare(msgMask);

            std::lock_guard<std::mutex> lock(mask_mutex_);
            latest_mask_ = cvMask->image.clone();
            mask_received_ = true;  
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception Mask!" << std::endl;
        }
    }

    void RgbdSlamNode::DetectionsCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
    {
        // =========================================
        // THREAD-SAFE CACHE
        // =========================================
        {
            std::lock_guard<std::mutex> lock(detections_mutex_);

            latest_detections_ = *msg;
        }

        detections_received_ = true;

        // =========================================
        // OPTIONAL DEBUG PRINT
        // =========================================
        // for(const auto& det : msg->detections)
        // {
        //     float cx = det.bbox.center.position.x;
        //     float cy = det.bbox.center.position.y;

        //     float w = det.bbox.size_x;
        //     float h = det.bbox.size_y;

        //     int x1 = static_cast<int>(cx - w * 0.5f);
        //     int y1 = static_cast<int>(cy - h * 0.5f);

        //     int x2 = static_cast<int>(cx + w * 0.5f);
        //     int y2 = static_cast<int>(cy + h * 0.5f);

        //     std::string class_id = "unknown";
        //     float confidence = 0.0f;

        //     if(!det.results.empty())
        //     {
        //         class_id = det.results[0].hypothesis.class_id;
        //         confidence = det.results[0].hypothesis.score;
        //     }

        //     std::cout
        //         << "Detection: "
        //         << class_id
        //         << " conf: " << confidence
        //         << " bbox: ["
        //         << x1 << ", "
        //         << y1 << ", "
        //         << x2 << ", "
        //         << y2 << "]"
        //         << std::endl;
        // }
    }
}
