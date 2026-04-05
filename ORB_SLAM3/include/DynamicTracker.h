#pragma once
#include "DynamicObject.h"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include "Frame.h"
#include "Hungarian.h"
#include <utility>
#include <unordered_map>
#include <limits>
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

struct DynamicPtsInfo {
        std::vector<cv::Point3f> prevPoints3D;
        std::vector<cv::Point3f> currPoints3D;
        //std::vector<size_t> dynamicGrid;
        //std::vector<std::pair<int,int>> dynamicPos;
        std::vector<std::pair<int,int>> dynamicMatchesIndex;
};

class DynamicTracker {
public:
    std::vector<DynamicObject> CurrentObjects;
    std::vector<DynamicObject> PrevObjects;
    DynamicPtsInfo dynamic_info;
    std::vector<std::pair<cv::Point3f, cv::Point3f>> optical_flow_matches;
    std::vector<std::pair<int,int>> idx_matches;
    int next_id;
    const float DIST_THRESH   = 1.0f;
    const float MOTION_THRESH = 0.8f;

    DynamicTracker();

    bool isSamePoint(const cv::Point3f& a, const cv::Point3f& b, float eps = 1e-3f)
    {
        return cv::norm(a - b) < eps;
    }
    
    void ProcessFrame(
        Frame& mCurrentFrame,
        Frame& mLastFrame,
        const std::vector<std::pair<int,int>>& _idx_matches, 
        const std::vector<std::pair<cv::Point3f,cv::Point3f>>& _optical_flow_matches
    );

    void ProcessFrame(
        Frame& mCurrentFrame,
        Frame& mLastFrame,
        const std::vector<cv::Point3f>& _currPoints,
        const std::vector<cv::Point3f>& _prevPoints,
        const std::vector<std::pair<int,int>>& _dynamicMatchesIndex
    );

private:
    Hungarian hungarian_solver;

    void rstVars(); 
    
    std::vector<std::vector<int>> ClusterPoints2(
        Frame& mCurrentFrame,
        Frame& mLastFrame   
    );

    void ClusterPoints(
        Frame& mCurrentFrame,
        const GridType& Grid,
        const std::vector<std::pair<int,int>>& GridPos,
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