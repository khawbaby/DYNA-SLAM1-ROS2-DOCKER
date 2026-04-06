#include "DynamicObject.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
namespace ORB_SLAM3
{

DynamicObject::DynamicObject(int _id): id(_id), age(1), isActive(true)
{
    R = cv::Mat::eye(3,3,CV_32F);
    t = cv::Mat::zeros(3,1,CV_32F);
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

void DynamicObject::Update(const std::vector<cv::Point3f>& newPoints)
{
    points3D = newPoints;
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
}

void DynamicObject::FitEllipsoid()
{
    if(points3D.size() < 5) return;

    cv::Mat mean = cv::Mat::zeros(3,1,CV_32F);

    for(auto &p : points3D)
    {
        mean.at<float>(0) += p.x;
        mean.at<float>(1) += p.y;
        mean.at<float>(2) += p.z;
    }

    mean /= (float)points3D.size();

    cv::Mat cov = cv::Mat::zeros(3,3,CV_32F);

    for(auto &p : points3D)
    {
        cv::Mat pt = (cv::Mat_<float>(3,1) << p.x, p.y, p.z);
        // diff = (3 x 1)
        cv::Mat diff = pt - mean;

        // cov = (3x1) * (1x3)
        cov += diff * diff.t();
    }

    // point3D is (3x1)
    cov /= (float)points3D.size();
    //std::cout << "cov: " << cov << std::endl;
    // Eigen decomposition
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);
    bool ok = cv::eigen(cov, eigenvalues, eigenvectors);

    if(!ok)
    {
        std::cout << "Eigen decomposition FAILED" << std::endl;
    }

    if(eigenvalues.empty())
    {
        std::cout << "Eigenvalues is EMPTY" << std::endl;
    }

    

    //std::cout << "eigenvalues: " << eigenvalues << std::endl;
    
    // axes is (3x1) How stretched it is along each axis
    axes = eigenvalues.clone();        // size
    
    // eigenvectors (3x3) How much it is pointing towards each axis
    orientation = eigenvectors.clone(); // rotation
    center = mean.clone();

    // Convert variance → actual size
    cv::sqrt(axes, axes);
    
    const int steps = 20;

    float a = axes.at<float>(0,0);
    float b = axes.at<float>(1,0);
    float c = axes.at<float>(2,0);

    for(int i = 0; i < steps; i++)
    {
        float theta = CV_PI * i / steps; // 0 → π

        for(int j = 0; j < steps; j++)
        {
            float phi = 2 * CV_PI * j / steps; // 0 → 2π

            // Unit sphere
            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);

            // Scale → ellipsoid
            cv::Mat pt = (cv::Mat_<float>(3,1) << a*x, b*y, c*z);

            // Rotate
            pt = orientation * pt;

            // Translate
            pt += center;

            ellipsoidPoints.emplace_back(
                pt.at<float>(0),
                pt.at<float>(1),
                pt.at<float>(2)
            );
        }
    }
    
}

void DynamicObject::DrawEllipsoid2D(
    cv::Mat &image,
    const cv::Mat &K // camera intrinsic matrix
)
{
    for(const auto &p : ellipsoidPoints)
    {
        cv::Mat pt3D = (cv::Mat_<float>(3,1) << p.x, p.y, p.z);

        // Project to 2D: x' = K * X
        cv::Mat proj = K * pt3D;

        float u = proj.at<float>(0) / proj.at<float>(2);
        float v = proj.at<float>(1) / proj.at<float>(2);

        if(u >= 0 && u < image.cols &&
           v >= 0 && v < image.rows)
        {
            cv::circle(image, cv::Point(u,v), 1, cv::Scalar(0,255,0), -1);
        }
    }
}



}