/**
 * @file rgbd-slam-node.cpp
 * @brief Implementation of the RgbdSlamNode Wrapper class.
 * @author Suchetan R S (rssuchetan@gmail.com)
 */
#include "rgbd-slam-node.hpp"

#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

namespace ORB_SLAM3_Wrapper
{
    using namespace WrapperTypeConversions;
    RgbdSlamNode::RgbdSlamNode(const std::string &strVocFile,
                               const std::string &strSettingsFile,
                               ORB_SLAM3::System::eSensor sensor)
        : SlamNodeBase("ORB_SLAM3_RGBD_ROS2", strVocFile, strSettingsFile, sensor)
    {
        // Declare parameters (topic names)
        this->declare_parameter("rgb_image_topic_name", rclcpp::ParameterValue("camera/image_raw"));
        this->declare_parameter("depth_image_topic_name", rclcpp::ParameterValue("depth/image_raw"));
        this->declare_parameter("dynamic_mask_topic_name", rclcpp::ParameterValue("yolo/dynamic_mask"));
        
        // Synced ROS Subscribers
        rgbSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("rgb_image_topic_name").as_string());
        depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->get_parameter("depth_image_topic_name").as_string());
        
        // Async Mask Subscriber
        maskSub_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("dynamic_mask_topic_name").as_string(),
            10,
            std::bind(&RgbdSlamNode::MaskCallback, this, std::placeholders::_1)
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
        // Copy the ros rgb image message to cv::Mat.
        try
        {
            cvRGB = cv_bridge::toCvShare(msgRGB);
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception RGB!" << endl;
            return;
        }

        // Copy the ros depth image message to cv::Mat.
        try
        {
            cvD = cv_bridge::toCvShare(msgD);
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception D!" << endl;
            return;
        }

        // ===== GET LATEST MASK (THREAD SAFE) =====
        cv::Mat mask_copy;

        {
            std::lock_guard<std::mutex> lock(mask_mutex_);
            mask_copy = latest_mask_.clone();
        }

        // ===== HANDLE EMPTY MASK =====
        if (mask_copy.empty()) {
            mask_copy = cv::Mat::ones(cvRGB->image.size(), CV_8UC1);
        }
    
        // track the frame.
        auto Tcw = interface()->slam()->TrackRGBD(cvRGB->image, cvD->image, mask_copy, stampToSec(msgRGB->header.stamp));
        
        // process the tracked pose.
        if (interface()->processTrackedPose(Tcw))
        {
            // Use RGB timestamp as the source stamp for TF/pose publishing.
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
        }
        catch (cv_bridge::Exception &e)
        {
            std::cerr << "cv_bridge exception Mask!" << std::endl;
        }
    }
}
