#pragma once
#include "DynamicObject.h"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include "Frame.h"
#include "Hungarian.h"

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
using GridType = std::vector<std::size_t>[FRAME_GRID_COLS][FRAME_GRID_ROWS];

class DynamicTracker {
public:
    std::vector<DynamicObject> CurrentObjects;
    std::vector<DynamicObject> PrevObjects;
    int next_id;

    DynamicTracker();

    void ProcessFrame(
        Frame& mCurrentFrame,
        Frame& mLastFrame
    );

private:
    Hungarian hungarian_solver;
    
    void ClusterPoints(
        const std::vector<cv::Point3f>& currPoints,
        const std::vector<cv::Point3f>& prevPoints,
        std::vector<std::vector<int>>& clusters
    );

    void ClusterPoints(
    const std::vector<cv::Point3f>& currPoints,
    const std::vector<cv::Point3f>& prevPoints,
    const GridType& dynamicGrid,
    const std::vector<std::pair<int,int>>& dynamicGridPos,
    std::vector<std::vector<int>>& clusters
    );

    float computeSigma(const std::vector<float>& data);
    //float computeCost(const Object& track, const Object& det,
                  //float sigma_d, float sigma_m, float sigma_s); 

    int greedyMatching(DynamicObject& newObj);
    //int HungarianMatching(DynamicObject& newObj);
    float Distance(const cv::Point3f& a, const cv::Point3f& b);
};

// std::vector<std::size_t> mDynamicGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
}