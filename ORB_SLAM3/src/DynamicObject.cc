#include "DynamicObject.h"
#include <opencv2/core.hpp>
#include <iostream>

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
    
    // cv::Mat test = (cv::Mat_<float>(3,1) << 0.5, 0.2, 0.5);
    // test = cv::norm (axes - test);
    // std::cout << "Euclidean Distance : " << test << std::endl;
}
}