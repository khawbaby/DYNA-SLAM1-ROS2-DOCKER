#include "DynamicObject.h"
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

DynamicObject::DynamicObject(int _id): id(_id), age(1), isActive(true)
{
    R = cv::Mat::eye(3,3,CV_32F);
    //t = cv::Mat::zeros(3,1,CV_32F);
}

float DynamicObject::ComputeIoU(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);

    int x2 = std::min(a.x + a.width,
                      b.x + b.width);

    int y2 = std::min(a.y + a.height,
                      b.y + b.height);

    int interArea =
        std::max(0, x2 - x1) *
        std::max(0, y2 - y1);

    int unionArea =
        a.area() + b.area() - interArea;

    if(unionArea <= 0)
        return 0.0f;

    return static_cast<float>(interArea) / unionArea;
}

void DynamicObject::Update(const std::vector<cv::Point3f>& newPoints,
                           const std::vector<cv::Point3f>& prevPoints)
{
    points3D = newPoints;
    prevPoints3D = prevPoints;
    age++;
    
    if (prevCentroid3D.x == 0 && prevCentroid3D.y == 0 && prevCentroid3D.z == 0) {
        cv::Point3f centroid3D(0.0f, 0.0f, 0.0f);
    }
}

void DynamicObject::Update(const std::vector<cv::Point3f>& newPoints, const std::vector<cv::KeyPoint>& newPoints2D)
{
    points3D = newPoints;
    points2D = newPoints2D;
    if(points2D.empty()) {
        has2DObservation = false; 
    } else {
        has2DObservation = true;
    }
}
// void DynamicObject::ComputeCentroid()
// {
//     prevCentroid3D = centroid3D;
//     cv::Point3f c(0,0,0);
//     for(auto &p : points3D)
//         c += p;

//     if(!points3D.empty())
//         c *= (1.0f / points3D.size());

//     centroid3D = c;
//     if (prevCentroid3D.x == 0 && prevCentroid3D.y == 0 && prevCentroid3D.z == 0) {
//         prevCentroid3D = centroid3D;
//     }
// }

void DynamicObject::ComputeCentroid()
{

    cv::Point3f c(0,0,0);
    for(auto &p : points3D)
        c += p;

    if(!points3D.empty())
        c *= (1.0f / points3D.size());

    centroid3D = c;

    cv::Point2f c2(0,0);

    for(const auto& kp : points2D)
        c2 += kp.pt;

    if(!points2D.empty())
        c2 *= (1.0f / points2D.size());

    centroid2D = c2;
}

void DynamicObject::FitEllipsoid()
{
    if(points3D.size() < 5) return;

    // --- Mean ---
    cv::Mat mean = cv::Mat::zeros(3,1,CV_32F);
    for(auto &p : points3D)
    {
        mean.at<float>(0) += p.x;
        mean.at<float>(1) += p.y;
        mean.at<float>(2) += p.z;
    }
    mean /= (float)points3D.size();
    
    // --- Covariance ---
    cv::Mat cov = cv::Mat::zeros(3,3,CV_32F);
    for(auto &p : points3D)
    {
        cv::Mat pt = (cv::Mat_<float>(3,1) << p.x, p.y, p.z);
        cv::Mat diff = pt - mean;
        cov += diff * diff.t();
    }
    cov /= (float)points3D.size();

    // --- Eigen decomposition ---
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    axes = eigenvalues.clone();
    orientation = eigenvectors.clone();
    center = mean.clone();

    // Convert variance → scale
    cv::sqrt(axes, axes);

    // --- Build SE3 pose ---
    cv::Mat R_cv = orientation.t();  // object → world

    // use DOUBLE from the start
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R_cv, R_eigen);

    // --- SVD orthogonalization ---
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        R_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV
    );

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Proper rotation projection
    Eigen::Matrix3d R_fixed = U * V.transpose();

    // Ensure right-handed system
    if(R_fixed.determinant() < 0)
    {
        U.col(2) *= -1;
        R_fixed = U * V.transpose();
    }

    // --- translation (double directly) ---
    Eigen::Vector3d t_d(
        center.at<float>(0),
        center.at<float>(1),
        center.at<float>(2)
    );

    // --- Final SE3 ---
    T_obj = Sophus::SE3d(R_fixed, t_d);
    
    // --- Generate LOCAL ellipsoid points ONLY ---
    ellipsoidPointsLocal.clear();

    const int steps = 10;  // reduce for performance

    float a = axes.at<float>(0,0);
    float b = axes.at<float>(1,0);
    float c = axes.at<float>(2,0);

    for(int i = 0; i < steps; i++)
    {
        float theta = CV_PI * i / steps;

        for(int j = 0; j < steps; j++)
        {
            float phi = 2 * CV_PI * j / steps;

            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);

            // LOCAL frame (NO rotation, NO translation)
            ellipsoidPointsLocal.emplace_back(
                a * x,
                b * y,
                c * z
            );
        }
    }
}

void DynamicObject::DrawEllipsoid2D(
    cv::Mat &image,
    const cv::Mat &K,
    const Sophus::SE3<float> &Tcw   // ADD THIS
)
{
    double fx = K.type() == CV_32F ? K.at<float>(0,0) : K.at<double>(0,0);
    double fy = K.type() == CV_32F ? K.at<float>(1,1) : K.at<double>(1,1);
    double cx = K.type() == CV_32F ? K.at<float>(0,2) : K.at<double>(0,2);
    double cy = K.type() == CV_32F ? K.at<float>(1,2) : K.at<double>(1,2);

    for(const auto &p : ellipsoidPointsLocal)
    {
        // LOCAL → Eigen
        // local → double
        Eigen::Vector3d pt_local(p.x, p.y, p.z);

        // object transform
        Eigen::Vector3d pt_world = T_obj * pt_local;

        // camera transform (convert ONCE)
        Sophus::SE3d Tcw_d = Tcw.cast<double>();

        Eigen::Vector3d pt_cam = Tcw_d * pt_world;

        if(pt_cam.z() <= 0) continue;

        // CAMERA → IMAGE
        double u = fx * pt_cam.x() / pt_cam.z() + cx;
        double v = fy * pt_cam.y() / pt_cam.z() + cy;

        if(u >= 0 && u < image.cols &&
           v >= 0 && v < image.rows)
        {
            cv::circle(image, cv::Point(u,v), 1,
                       cv::Scalar(0,255,0), -1);
        }
    }
}

void DynamicObject::UpdatePoseFromState()
{
    // --- Build rotation from stored orientation ---
    cv::Mat R_cv = orientation.t();  // same as before

    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R_cv, R_eigen);

    // Orthogonalize (same as your code)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        R_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV
    );

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R_fixed = U * V.transpose();

    if(R_fixed.determinant() < 0)
    {
        U.col(2) *= -1;
        R_fixed = U * V.transpose();
    }

    // --- Translation from centroid ---
    Eigen::Vector3d t(
        centroid3D.x,
        centroid3D.y,
        centroid3D.z
    );

    T_obj = Sophus::SE3d(R_fixed, t);

    ellipsoidPointsLocal.clear();

    const int steps = 10;  // reduce for performance

    float a = axes.at<float>(0,0);
    float b = axes.at<float>(1,0);
    float c = axes.at<float>(2,0);

    for(int i = 0; i < steps; i++)
    {
        float theta = CV_PI * i / steps;

        for(int j = 0; j < steps; j++)
        {
            float phi = 2 * CV_PI * j / steps;

            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);

            // LOCAL frame (NO rotation, NO translation)
            ellipsoidPointsLocal.emplace_back(
                a * x,
                b * y,
                c * z
            );
        }
    }
}

void DynamicObject::UpdateFromMeasurement(
    const DynamicObject& meas,
    const DynamicObject& prev)
{
    // --- Initialize if needed ---
    if(!kf_initialized)
    {
        centroid3D = meas.centroid3D;
        InitKalman();
    }

    // =========================
    // 1. PREDICTION
    // =========================
    Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
    F(0,3) = 1; F(1,4) = 1; F(2,5) = 1;

    Eigen::Matrix<float,6,6> Q = Eigen::Matrix<float,6,6>::Identity() * 0.01f;

    kf_x = F * kf_x;
    kf_P = F * kf_P * F.transpose() + Q;

    // =========================
    // 2. MEASUREMENT
    // =========================
    Eigen::Matrix<float,3,6> H;
    H.setZero();
    H(0,0)=1; H(1,1)=1; H(2,2)=1;

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    R(0,0) = 0.05f;
    R(1,1) = 0.05f;
    R(2,2) = 0.5f;   // 🔥 depth noisy

    Eigen::Vector3f z;
    z << meas.centroid3D.x,
         meas.centroid3D.y,
         meas.centroid3D.z;

    // =========================
    // 3. UPDATE
    // =========================
    Eigen::Vector3f y = z - H * kf_x;

    Eigen::Matrix3f S = H * kf_P * H.transpose() + R;
    Eigen::Matrix<float,6,3> K = kf_P * H.transpose() * S.inverse();

    kf_x = kf_x + K * y;
    kf_P = (Eigen::Matrix<float,6,6>::Identity() - K * H) * kf_P;

    // =========================
    // 4. WRITE BACK
    // =========================
    centroid3D.x = kf_x(0);
    centroid3D.y = kf_x(1);
    centroid3D.z = kf_x(2);

    velocity.x = kf_x(3);
    velocity.y = kf_x(4);
    velocity.z = kf_x(5);

    // =========================
    // 5. AXES (keep smoothing)
    // =========================
    axes = prev.axes.clone();

    for(int k = 0; k < 3; k++)
    {
        float prev_val = prev.axes.at<float>(k);
        float meas_val = meas.axes.at<float>(k);

        float blended = 0.3f * meas_val + 0.7f * prev_val;

        float max_change = 0.2f * prev_val;
        float diff = blended - prev_val;

        if(std::abs(diff) > max_change)
            blended = prev_val + std::copysign(max_change, diff);

        axes.at<float>(k) = blended;
    }

    // =========================
    // 6. ORIENTATION (freeze)
    // =========================
    orientation = prev.orientation.clone();

    // =========================
    // 7. UPDATE POSE
    // =========================
    UpdatePoseFromState();
}


}