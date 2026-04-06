#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
class DynamicObject {
public:
    int id;

    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point3f> prevPoints3D;
    std::vector<cv::Point3f> ellipsoidPoints;

    cv::Mat R;  // rotation
    cv::Mat t;  // translation
    cv::Mat axes;        // 3x1
    cv::Mat orientation; // 3x3
    cv::Mat center;      // 3x1 (optional, same as centroid)

    cv::Point3f centroid3D;
    cv::Point3f prevCentroid3D = cv::Point3f(-1.0f, -1.0f, -1.0f); 

    cv::Point3d axes3D;

    int age;    
    bool isActive;

    DynamicObject(int _id);

    void Update(const std::vector<cv::Point3f>& newPoints,
                const std::vector<cv::Point3f>& prevPoints);

    void Update(const std::vector<cv::Point3f>& newPoints);
    void ComputeCentroid();
    void FitEllipsoid();

    void DrawEllipsoid2D(
        cv::Mat &image,
        const cv::Mat &K // camera intrinsic matrix
    );

};

}