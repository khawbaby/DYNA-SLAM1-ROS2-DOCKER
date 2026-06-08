#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "sensor_msgs/image_encodings.hpp"
class CameraNode : public rclcpp::Node
{
public:
    CameraNode() : Node("camera_node")
    {
        // RGB subscriber
        color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&CameraNode::colorCallback, this, std::placeholders::_1));

        // Depth subscriber
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_rect_raw", 10,
            std::bind(&CameraNode::depthCallback, this, std::placeholders::_1));

        cv::namedWindow("RGB", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);

        RCLCPP_INFO(this->get_logger(), "Camera node started");
    }

private:
    void colorCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

            cv::imshow("RGB", frame);
            cv::waitKey(1);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            cv::Mat depth = cv_bridge::toCvCopy(
                msg,
                sensor_msgs::image_encodings::TYPE_16UC1
            )->image;

            cv::Mat depth_vis;
            cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

            cv::imshow("Depth", depth_vis);
            cv::waitKey(1);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        }
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraNode>());
    rclcpp::shutdown();

    cv::destroyAllWindows();
    return 0;
}