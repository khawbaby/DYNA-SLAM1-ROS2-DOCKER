#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include "sophus/se3.hpp"

namespace ORB_SLAM3
{
class DynamicObject {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id;
    int missed_frames = 0;
    
    // --- Kalman state ---
    Eigen::Matrix<float,6,1> kf_x;   // [x y z vx vy vz]
    Eigen::Matrix<float,6,6> kf_P;   // covariance
    bool kf_initialized = false;

    Eigen::Matrix3f R_eigen;
    Sophus::SE3<double> T_obj;

    std::vector<Eigen::Vector3d> points3D_local;
    std::vector<cv::KeyPoint> points2D;
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point3f> prevPoints3D;
    std::vector<cv::Point3f> ellipsoidPoints;
    std::vector<cv::Point3f> ellipsoidPointsLocal;

    cv::Rect bbox;

    cv::Mat R;  // rotation
    cv::Mat axes;        // 3x1
    cv::Mat orientation; // 3x3
    cv::Mat center;      // 3x1 (optional, same as centroid)
    cv::Vec3f t; // translation

    cv::Point2f centroid2D;
    cv::Point2f velocity2D;

    cv::Point3f centroid3D;
    cv::Point3f prevCentroid3D = cv::Point3f(-1.0f, -1.0f, -1.0f); 
    cv::Point3f velocity = cv::Point3f(-1000.0f, -1.0f, -1.0f);
    cv::Point3d axes3D;

    bool has2DObservation;
    int age;    
    bool isActive;
    
public:
    DynamicObject(int _id);

    float ComputeIoU(const cv::Rect& a, const cv::Rect& b);

    void Update(const std::vector<cv::Point3f>& newPoints,
                const std::vector<cv::Point3f>& prevPoints);

    void Update(const std::vector<cv::Point3f>& newPoints, const std::vector<cv::KeyPoint>& newPoints2D);
    void ComputeCentroid();
    void FitEllipsoid();

    void DrawEllipsoid2D(
        cv::Mat &image,
        const cv::Mat &K, // camera intrinsic matrix
        const Sophus::SE3<float> &Tcw  
    );

    void UpdateFromMeasurement(
        const DynamicObject& meas,
        const DynamicObject& prev
    );

    void UpdatePoseFromState();

    void InitKalman()
    {
        kf_x << centroid3D.x, centroid3D.y, centroid3D.z, 0, 0, 0;
        kf_P = Eigen::Matrix<float,6,6>::Identity() * 0.1f;
        kf_initialized = true;
    }

    
};

}