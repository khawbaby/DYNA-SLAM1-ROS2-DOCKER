#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

struct Detection
{
    cv::Rect bbox;

    int class_id;

    float confidence;
};

}

#endif