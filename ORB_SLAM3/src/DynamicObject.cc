#include "DynamicObject.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "sophus/se3.hpp"

namespace ORB_SLAM3
{

DynamicObject::DynamicObject(int _id)
    : id(_id)
    , age(1)
    , isActive(true)
    , missed_frames(0)
    , tracked_frames(0)
    , kf_initialized(false)
    , has2DObservation(false)
    , velocity(-1000.0f, -1000.0f, -1000.0f)
{
    R = cv::Mat::eye(3, 3, CV_32F);
}


DynamicObject::DynamicObject(const DynamicObject& other)
    : id(other.id)
    , age(other.age)
    , isActive(other.isActive)
    , missed_frames(other.missed_frames)
    , tracked_frames(other.tracked_frames)
    , kf_x(other.kf_x)
    , kf_P(other.kf_P)
    , kf_initialized(other.kf_initialized)
    , R_eigen(other.R_eigen)
    , T_obj(other.T_obj)
    , points3D_local(other.points3D_local)
    , points2D(other.points2D)
    , points3D(other.points3D)
    , prevPoints3D(other.prevPoints3D)
    , ellipsoidPoints(other.ellipsoidPoints)
    , ellipsoidPointsLocal(other.ellipsoidPointsLocal)
    , bbox(other.bbox)
    , t(other.t)
    , centroid2D(other.centroid2D)
    , velocity2D(other.velocity2D)
    , centroid3D(other.centroid3D)
    , prevCentroid3D(other.prevCentroid3D)
    , velocity(other.velocity)
    , axes3D(other.axes3D)
    , has2DObservation(other.has2DObservation)
    , pointsHistoryBuffer(other.pointsHistoryBuffer)
    , prevOrientEigen(other.prevOrientEigen)
    , hasPrevOrient(other.hasPrevOrient)
{
    R           = other.R.clone();
    axes        = other.axes.clone();
    orientation = other.orientation.clone();
    center      = other.center.clone();
}

DynamicObject& DynamicObject::operator=(const DynamicObject& other)
{
    if(this == &other) return *this;

    id               = other.id;
    age              = other.age;
    isActive         = other.isActive;
    missed_frames    = other.missed_frames;
    tracked_frames   = other.tracked_frames;
    kf_x             = other.kf_x;
    kf_P             = other.kf_P;
    kf_initialized   = other.kf_initialized;
    R_eigen          = other.R_eigen;
    T_obj            = other.T_obj;
    points3D_local   = other.points3D_local;
    points2D         = other.points2D;
    points3D         = other.points3D;
    prevPoints3D     = other.prevPoints3D;
    ellipsoidPoints  = other.ellipsoidPoints;
    ellipsoidPointsLocal = other.ellipsoidPointsLocal;
    bbox             = other.bbox;
    t                = other.t;
    centroid2D       = other.centroid2D;
    velocity2D       = other.velocity2D;
    centroid3D       = other.centroid3D;
    prevCentroid3D   = other.prevCentroid3D;
    velocity         = other.velocity;
    axes3D           = other.axes3D;
    has2DObservation = other.has2DObservation;
    pointsHistoryBuffer = other.pointsHistoryBuffer;
    prevOrientEigen  = other.prevOrientEigen;
    hasPrevOrient    = other.hasPrevOrient;

    R           = other.R.clone();
    axes        = other.axes.clone();
    orientation = other.orientation.clone();
    center      = other.center.clone();

    return *this;
}

// ----------------------------------------------------------------
// KALMAN FILTER
// ----------------------------------------------------------------
void DynamicObject::UpdateKalmanFilter(const cv::Point3f& measuredCentroid)
{
    if(!kf_initialized)
    {
        kf_x.setZero();
        kf_x(0) = measuredCentroid.x;
        kf_x(1) = measuredCentroid.y;
        kf_x(2) = measuredCentroid.z;

        kf_P = Eigen::Matrix<float,6,6>::Identity();
        kf_initialized = true;

        centroid3D = measuredCentroid;
        velocity   = cv::Point3f(0.0f, 0.0f, 0.0f);
        return;
    }

    // --- Predict ---
    Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
    F(0,3) = 1; F(1,4) = 1; F(2,5) = 1;

    // Tighter process noise on position (driven by velocity), looser on
    // velocity (can accelerate freely between frames at ~30 Hz).
    Eigen::Matrix<float,6,6> Q = Eigen::Matrix<float,6,6>::Zero();
    Q(0,0) = 0.001f; Q(1,1) = 0.001f; Q(2,2) = 0.001f;
    Q(3,3) = 0.05f;  Q(4,4) = 0.05f;  Q(5,5) = 0.05f;

    kf_x = F * kf_x;
    kf_P = F * kf_P * F.transpose() + Q;

    // --- Update ---
    Eigen::Matrix<float,3,6> H = Eigen::Matrix<float,3,6>::Zero();
    H(0,0) = 1; H(1,1) = 1; H(2,2) = 1;

    // Depth (Z) is noisier than X/Y for RGB-D
    Eigen::Matrix3f Rn = Eigen::Matrix3f::Identity();
    Rn(0,0) = 0.05f;
    Rn(1,1) = 0.05f;
    Rn(2,2) = 0.5f;

    Eigen::Vector3f z;
    z << measuredCentroid.x, measuredCentroid.y, measuredCentroid.z;

    Eigen::Vector3f          innov = z - H * kf_x;
    Eigen::Matrix3f          S     = H * kf_P * H.transpose() + Rn;
    Eigen::Matrix<float,6,3> K     = kf_P * H.transpose() * S.inverse();

    kf_x = kf_x + K * innov;
    kf_P = (Eigen::Matrix<float,6,6>::Identity() - K * H) * kf_P;

    // --- Write back ---
    centroid3D.x = kf_x(0);
    centroid3D.y = kf_x(1);
    centroid3D.z = kf_x(2);

    velocity.x = kf_x(3);
    velocity.y = kf_x(4);
    velocity.z = kf_x(5);
}

// ----------------------------------------------------------------
// UPDATE FROM MEASUREMENT (called every matched frame)
// ----------------------------------------------------------------
void DynamicObject::UpdateFromMeasurement(const DynamicObject& meas, const DynamicObject& prev)
{
    // Geometry from measurement
    points3D = meas.points3D;
    points2D = meas.points2D;
    bbox     = meas.bbox;

    // Merge: start from prev's optical-flow history, then append the dense
    // depth samples that SampleDepthPoints stored in meas before this call
    pointsHistoryBuffer = prev.pointsHistoryBuffer;
    pointsHistoryBuffer.insert(pointsHistoryBuffer.end(),
        meas.pointsHistoryBuffer.begin(),
        meas.pointsHistoryBuffer.end());
    while((int)pointsHistoryBuffer.size() > MAX_HISTORY_POINTS)
        pointsHistoryBuffer.erase(pointsHistoryBuffer.begin());

    prevOrientEigen = prev.prevOrientEigen;
    hasPrevOrient   = prev.hasPrevOrient;

    // Refit with accumulated history — produces stable orientation + axes
    FitEllipsoid();

    // Smooth axes magnitude (shape changes slowly)
    if(!prev.axes.empty() && !axes.empty())
    {
        for(int k = 0; k < 3; k++)
        {
            float prev_val = prev.axes.at<float>(k);
            float meas_val = axes.at<float>(k);
            float blended  = 0.3f * meas_val + 0.7f * prev_val;

            float max_change = 0.2f * std::max(prev_val, 1e-4f);
            float diff       = blended - prev_val;
            if(std::abs(diff) > max_change)
                blended = prev_val + std::copysign(max_change, diff);

            axes.at<float>(k) = blended;
        }
    }

    // KF update — smooths centroid and estimates velocity
    UpdateKalmanFilter(meas.centroid3D);

    // Counters
    missed_frames  = 0;
    tracked_frames = prev.tracked_frames + 1;

    // Rebuild T_obj with KF-refined centroid + stable orientation from FitEllipsoid
    UpdatePoseFromState();
}

// ----------------------------------------------------------------
// COMPUTE CENTROID
// ----------------------------------------------------------------
void DynamicObject::ComputeCentroid()
{
    cv::Point3f c(0, 0, 0);
    for(auto& p : points3D) c += p;
    if(!points3D.empty()) c *= (1.0f / (float)points3D.size());
    centroid3D = c;

    cv::Point2f c2(0, 0);
    for(const auto& kp : points2D) c2 += kp.pt;
    if(!points2D.empty()) c2 *= (1.0f / (float)points2D.size());
    centroid2D = c2;
}

// ----------------------------------------------------------------
// UPDATE (raw points, no KF)
// ----------------------------------------------------------------
void DynamicObject::Update(const std::vector<cv::Point3f>& newPoints,
                           const std::vector<cv::Point3f>& prevPoints)
{
    points3D     = newPoints;
    prevPoints3D = prevPoints;
    age++;
}

void DynamicObject::Update(const std::vector<cv::Point3f>& newPoints,
                           const std::vector<cv::KeyPoint>& newPoints2D)
{
    points3D         = newPoints;
    points2D         = newPoints2D;
    has2DObservation = !points2D.empty();
}

// ----------------------------------------------------------------
// SAMPLE DENSE DEPTH POINTS FROM BOUNDING BOX
// Adds centroid-relative world-frame points directly into the history
// buffer without touching points3D (optical flow points).  The dense
// samples give FitEllipsoid 10-50× more data per frame for stable PCA.
// ----------------------------------------------------------------
void DynamicObject::SampleDepthPoints(
    const cv::Mat& imDepth,
    const Sophus::SE3<float>& Tcw,
    float fx_, float fy_, float cx_, float cy_,
    float maxDepth)
{
    if(imDepth.empty() || bbox.area() <= 0) return;

    // Reference depth: centroid projected into camera frame
    Eigen::Vector3f Xc_cen = Tcw * Eigen::Vector3f(centroid3D.x, centroid3D.y, centroid3D.z);
    float refZ = Xc_cen.z();
    if(refZ <= 0.1f) return;

    Sophus::SE3f Twc = Tcw.inverse();

    int x0 = std::max(bbox.x, 0);
    int y0 = std::max(bbox.y, 0);
    int x1 = std::min(bbox.x + bbox.width,  imDepth.cols - 1);
    int y1 = std::min(bbox.y + bbox.height, imDepth.rows - 1);

    const int STRIDE = 4;

    for(int v = y0; v <= y1; v += STRIDE)
    {
        for(int u = x0; u <= x1; u += STRIDE)
        {
            float d = imDepth.at<float>(v, u);
            if(d <= 0.1f || d > maxDepth) continue;
            // Reject background pixels more than 0.8 m behind the centroid
            if(std::abs(d - refZ) > 0.8f) continue;

            Eigen::Vector3f Xc((u - cx_) * d / fx_,
                               (v - cy_) * d / fy_,
                               d);
            Eigen::Vector3f Xw = Twc * Xc;

            pointsHistoryBuffer.emplace_back(
                Xw.x() - centroid3D.x,
                Xw.y() - centroid3D.y,
                Xw.z() - centroid3D.z
            );
        }
    }

    while((int)pointsHistoryBuffer.size() > MAX_HISTORY_POINTS)
        pointsHistoryBuffer.erase(pointsHistoryBuffer.begin());
}

// ----------------------------------------------------------------
// FIT ELLIPSOID
// Accumulates centroid-relative points across frames so PCA runs on a
// richer point cloud (150 pts max) instead of a single sparse frame.
// Eigenvector sign-flip correction prevents orientation from jumping
// when two eigenvalues are nearly equal.
// ----------------------------------------------------------------
void DynamicObject::FitEllipsoid()
{
    if(points3D.size() < 5) return;

    // Compute centroid of current frame
    cv::Point3f c(0, 0, 0);
    for(auto& p : points3D) c += p;
    c *= (1.0f / (float)points3D.size());

    // Append centroid-relative current points to history buffer
    for(auto& p : points3D)
        pointsHistoryBuffer.emplace_back(p.x - c.x, p.y - c.y, p.z - c.z);
    while((int)pointsHistoryBuffer.size() > MAX_HISTORY_POINTS)
        pointsHistoryBuffer.erase(pointsHistoryBuffer.begin());

    const auto& pts = pointsHistoryBuffer;
    if(pts.size() < 5) return;

    // Covariance on accumulated centroid-relative points
    cv::Mat cov = cv::Mat::zeros(3, 3, CV_32F);
    for(auto& p : pts)
    {
        cv::Mat pt = (cv::Mat_<float>(3,1) << p.x, p.y, p.z);
        cov += pt * pt.t();
    }
    cov /= (float)pts.size();

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);  // rows = eigenvectors, descending λ

    // Correct eigenvector sign flips vs previous frame
    if(hasPrevOrient)
    {
        for(int k = 0; k < 3; k++)
        {
            Eigen::Vector3d ev_prev(prevOrientEigen(k,0),
                                    prevOrientEigen(k,1),
                                    prevOrientEigen(k,2));
            Eigen::Vector3d ev_curr(eigenvectors.at<float>(k,0),
                                    eigenvectors.at<float>(k,1),
                                    eigenvectors.at<float>(k,2));
            if(ev_prev.dot(ev_curr) < 0)
                eigenvectors.row(k) *= -1;
        }
    }

    axes        = eigenvalues.clone();
    orientation = eigenvectors.clone();
    center      = (cv::Mat_<float>(3,1) << c.x, c.y, c.z);
    cv::sqrt(axes, axes);

    // Build rotation: eigenvectors.t() has principal axes as columns
    cv::Mat R_cv = orientation.t();
    Eigen::Matrix3d R_e;
    cv::cv2eigen(R_cv, R_e);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_e, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d R_fixed = U * V.transpose();
    if(R_fixed.determinant() < 0) { U.col(2) *= -1; R_fixed = U * V.transpose(); }

    // Store eigenvectors for next-frame flip detection
    for(int k = 0; k < 3; k++)
    {
        prevOrientEigen(k,0) = eigenvectors.at<float>(k,0);
        prevOrientEigen(k,1) = eigenvectors.at<float>(k,1);
        prevOrientEigen(k,2) = eigenvectors.at<float>(k,2);
    }
    hasPrevOrient = true;

    Eigen::Vector3d t_d(c.x, c.y, c.z);
    T_obj = Sophus::SE3d(R_fixed, t_d);

    RebuildEllipsoidPoints();
}

// ----------------------------------------------------------------
// REBUILD ELLIPSOID LOCAL POINTS (shared by FitEllipsoid and UpdatePoseFromState)
// ----------------------------------------------------------------
void DynamicObject::RebuildEllipsoidPoints()
{
    ellipsoidPointsLocal.clear();

    if(axes.empty()) return;

    float a = axes.at<float>(0, 0);
    float b = axes.at<float>(1, 0);
    float c = axes.at<float>(2, 0);

    const int steps = 10;

    for(int i = 0; i < steps; i++)
    {
        float theta = CV_PI * i / steps;
        for(int j = 0; j < steps; j++)
        {
            float phi = 2.0f * CV_PI * j / steps;

            ellipsoidPointsLocal.emplace_back(
                a * std::sin(theta) * std::cos(phi),
                b * std::sin(theta) * std::sin(phi),
                c * std::cos(theta)
            );
        }
    }
}

// ----------------------------------------------------------------
// UPDATE POSE FROM KALMAN STATE
// Rebuilds T_obj using the KF-refined centroid and the orientation
// computed by FitEllipsoid (now stable via accumulated PCA).
// ----------------------------------------------------------------
void DynamicObject::UpdatePoseFromState()
{
    if(orientation.empty()) return;

    cv::Mat R_cv = orientation.t();
    Eigen::Matrix3d R_e;
    cv::cv2eigen(R_cv, R_e);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_e, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d R_fixed = U * V.transpose();
    if(R_fixed.determinant() < 0) { U.col(2) *= -1; R_fixed = U * V.transpose(); }

    Eigen::Vector3d t(centroid3D.x, centroid3D.y, centroid3D.z);
    T_obj = Sophus::SE3d(R_fixed, t);

    RebuildEllipsoidPoints();
}

// ----------------------------------------------------------------
// DRAW ELLIPSOID
// ----------------------------------------------------------------
void DynamicObject::DrawEllipsoid2D(
    cv::Mat& image,
    const cv::Mat& K,
    const Sophus::SE3<float>& Tcw)
{
    double fx = K.type() == CV_32F ? K.at<float>(0,0) : K.at<double>(0,0);
    double fy = K.type() == CV_32F ? K.at<float>(1,1) : K.at<double>(1,1);
    double cx = K.type() == CV_32F ? K.at<float>(0,2) : K.at<double>(0,2);
    double cy = K.type() == CV_32F ? K.at<float>(1,2) : K.at<double>(1,2);

    Sophus::SE3d Tcw_d = Tcw.cast<double>();

    for(const auto& p : ellipsoidPointsLocal)
    {
        Eigen::Vector3d pt_local(p.x, p.y, p.z);
        Eigen::Vector3d pt_world = T_obj * pt_local;
        Eigen::Vector3d pt_cam   = Tcw_d * pt_world;

        if(pt_cam.z() <= 0) continue;

        double u = fx * pt_cam.x() / pt_cam.z() + cx;
        double v = fy * pt_cam.y() / pt_cam.z() + cy;

        if(u >= 0 && u < image.cols && v >= 0 && v < image.rows)
            cv::circle(image, cv::Point((int)u, (int)v), 1,
                       cv::Scalar(0, 255, 0), -1);
    }
}

// ----------------------------------------------------------------
// IoU HELPER
// ----------------------------------------------------------------
float DynamicObject::ComputeIoU(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width,  b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    if(unionArea <= 0) return 0.0f;
    return static_cast<float>(interArea) / unionArea;
}

} // namespace ORB_SLAM3    