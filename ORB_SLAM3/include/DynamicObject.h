#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <iostream>

namespace ORB_SLAM3
{
class DynamicObject {
public:
    int id;

    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point3f> prevPoints3D;

    cv::Mat R;  // rotation
    cv::Mat t;  // translation
    cv::Mat axes;        // 3x1
    cv::Mat orientation; // 3x3
    cv::Mat center;      // 3x1 (optional, same as centroid)
    cv::Point3f centroid3D;
    cv::Point3f prevCentroid3D;
    
    int age;    
    bool isActive;

    DynamicObject(int _id);

    void Update(const std::vector<cv::Point3f>& newPoints,
                const std::vector<cv::Point3f>& prevPoints);

    void ComputeCentroid();

    void FitEllipsoid();

};

}