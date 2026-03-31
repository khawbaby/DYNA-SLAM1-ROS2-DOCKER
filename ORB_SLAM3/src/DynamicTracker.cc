#include <queue>
#include <iostream>
#include <iomanip>
#include "DynamicTracker.h"
#include "Frame.h"
#include "Hungarian.h"

namespace ORB_SLAM3
{
using GridType = std::vector<std::size_t>[FRAME_GRID_COLS][FRAME_GRID_ROWS];
DynamicTracker::DynamicTracker() : next_id(0) {}

float DynamicTracker::computeSigma(const std::vector<float>& data)
{
    if(data.empty()) return 0.0f;

    // Mean
    float mean = 0.0f;
    for(float x : data)
        mean += x;
    mean /= data.size();

    // Variance
    float var = 0.0f;
    for(float x : data)
        var += (x - mean) * (x - mean);
    var /= data.size();   // use (size - 1) if you want unbiased

    return std::sqrt(var);
}

int DynamicTracker::greedyMatching(DynamicObject& newObj) {
    float bestDist = 1e9;
    float bestMotion = 1e9;
    float bestSizeDiff = 1e9;

    int bestIdxDist = -1;
    int bestIdxMotion = -1;
    int bestIdxSize = -1;
    int bestIdx = -1;
    
    float sum_motion = 0; 
    float sum_dist = 0;
    float sum_size = 0;

    std::vector<float> motion_vect ;
    std::vector<float> dist_vect ;
    std::vector<float> size_vect ;

    for(int i = 0; i < CurrentObjects.size(); i++)
    {
        float dist = cv::norm(newObj.centroid3D - CurrentObjects[i].centroid3D);
        cv::Point3f vel_old = CurrentObjects[i].centroid3D - CurrentObjects[i].prevCentroid3D;
        cv::Point3f predicted = CurrentObjects[i].centroid3D + vel_old;
        float motion = cv::norm(newObj.centroid3D - predicted);
        float sizeDiff = cv::norm(CurrentObjects[i].axes - newObj.axes);
        
        motion_vect.push_back(motion);
        dist_vect.push_back(dist);
        size_vect.push_back(sizeDiff);
    }

    float sigma_d = computeSigma(dist_vect);
    float sigma_m = computeSigma(motion_vect);
    float sigma_s = computeSigma(size_vect);
    
    if (sigma_d == 0 || sigma_m == 0 || sigma_m == 0) 
        return -1;
    float bestScore = FLT_MAX;
    float d_score;
    float m_score;
    float s_score;
    
    for(int i = 0; i < CurrentObjects.size(); i++)
    {   
        d_score = (dist_vect[i]*dist_vect[i])/(sigma_d*sigma_d + 1e-3);
        m_score = (motion_vect[i]*motion_vect[i])/(sigma_m*sigma_m + 1e-3);
        s_score = (size_vect[i]*size_vect[i])/(sigma_s*sigma_s + 1e-3);

        float totalScore = d_score + m_score + s_score;

        if(totalScore < bestScore)
        {
            bestScore = totalScore;
            bestIdx = i;
        }
    }

    if (bestIdx != -1 ) {
        std::cout 
            << "Best Index: " << bestIdx << "\n"
            << " Best Distance: " << d_score  << std::setw(20) 
            << " Best Motion: " << m_score  << std::setw(20) 
            << " Size Diff: " << s_score  << std::endl;
                
        std::cout
            << " sigma_d: " << sigma_d << std::setw(25)
            << " sigma_m: " << sigma_m << std::setw(25)
            << " sigma_s: " << sigma_s << std::endl;
        std::cout << std::endl;
    }
    return bestIdx;
}

// float DynamicTracker::computeCost(const Object& track, const Object& det,
//                   float sigma_d, float sigma_m, float sigma_s)
// {
//     float dist = cv::norm(track.centroid3D - det.centroid3D);

//     float motion = cv::norm(track.velocity - det.velocity);

//     float sizeDiff = std::abs(track.size - det.size);

//     float d_score = (dist * dist) / (sigma_d * sigma_d);
//     float m_score = (motion * motion) / (sigma_m * sigma_m);
//     float s_score = (sizeDiff * sizeDiff) / (sigma_s * sigma_s);

//     return d_score + m_score + s_score;
// }


void DynamicTracker::ProcessFrame(Frame& mCurrentFrame,Frame& mLastFrame)
{

    auto &currPoints = mCurrentFrame.mvDynamicPoints3D;
    auto &prevPoints = mLastFrame.mvDynamicPoints3D;
    auto &dynamicGrid = mCurrentFrame.mDynamicGrid;
    auto &dynamicPos = mCurrentFrame.mvDynamicGridPos;

    // Step 1: cluster indices
    std::vector<std::vector<int>> clusters;
    ClusterPoints(currPoints, prevPoints, dynamicGrid, dynamicPos, clusters);
    //std::cout << "Number of CurrentObjects created: " << clusters.size() << std::endl;
    
    // Clusters consists of vector of points
    for(const auto& cluster : clusters)
    {
        std::vector<cv::Point3f> objCurr;
        std::vector<cv::Point3f> objPrev;

        for(int idx : cluster)
        {
            objCurr.push_back(currPoints[idx]);
            objPrev.push_back(prevPoints[idx]);
        }

        DynamicObject obj(next_id++);
        obj.Update(objCurr, objPrev);
        obj.ComputeCentroid();
        obj.FitEllipsoid();
        CurrentObjects.push_back(obj);
    }
    mCurrentFrame.mDynamicObjects = CurrentObjects;
    PrevObjects = mLastFrame.mDynamicObjects;

    if (mLastFrame.mDynamicObjects.empty())
        std::cout << "LAST FRAME NO DYNAMIC OBJECT" << std::endl;

    // Compute for all pairs
    std::vector<float> dist_vect, motion_vect, size_vect;
    for(int i = 0; i < PrevObjects.size(); i++)
    {
        for(int j = 0; j < CurrentObjects.size(); j++)
        {
            float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);

            cv::Point3f vel_old = PrevObjects[i].centroid3D - PrevObjects[i].prevCentroid3D;
            cv::Point3f predicted = PrevObjects[i].centroid3D + vel_old;

            float motion = cv::norm(CurrentObjects[j].centroid3D - predicted);

            float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

            dist_vect.push_back(dist);
            motion_vect.push_back(motion);
            size_vect.push_back(sizeDiff);

            // std::cout
            //     << " dist: " << dist << std::setw(25)
            //     << " motion: " << motion << std::setw(25)
            //     << " sizeDiff: " << sizeDiff << std::endl;
            // std::cout << std::endl;
        }
    }

    // Compute sigma
    float sigma_d = computeSigma(dist_vect);
    float sigma_m = computeSigma(motion_vect);
    float sigma_s = computeSigma(size_vect);
    
    sigma_d = std::max(sigma_d, 1e-3f);
    sigma_m = std::max(sigma_m, 1e-3f);
    sigma_s = std::max(sigma_s, 1e-3f);

    int N = PrevObjects.size();
    int M = CurrentObjects.size();

    std::vector<std::vector<float>> costMatrix(N, std::vector<float>(M, 1e6));

    // Fill up Cost Matrix
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);

            cv::Point3f vel_old = PrevObjects[i].centroid3D - PrevObjects[i].prevCentroid3D;
            cv::Point3f predicted = PrevObjects[i].centroid3D + vel_old;

            float motion = cv::norm(CurrentObjects[j].centroid3D - predicted);

            float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

            float d_score = (dist*dist)/(sigma_d*sigma_d);
            float m_score = (motion*motion)/(sigma_m*sigma_m);
            float s_score = (sizeDiff*sizeDiff)/(sigma_s*sigma_s);

            float totalScore = d_score + m_score + s_score;
            // std::cout
            //     << " dist: " << d_score << std::setw(25)
            //     << " motion: " << m_score << std::setw(25)
            //     << " sizeDiff: " << s_score << std::endl;
            // std::cout << std::endl;

            // Gating
            if(totalScore < 50.0f)
                costMatrix[i][j] = totalScore;
        }
    }
    
    //std::vector<int> assignment = hungarian_solver.solve(costMatrix);

    std::vector<bool> usedDetections(M, false);
    
}

float DynamicTracker::Distance(const cv::Point3f& a, const cv::Point3f& b)
{
    return cv::norm(a - b);
}

void DynamicTracker::ClusterPoints(const std::vector<cv::Point3f>& currPoints, const std::vector<cv::Point3f>& prevPoints,std::vector<std::vector<int>>& clusters)
{
    const float DIST_THRESH   = 0.5f;  // meters
    const float MOTION_THRESH = 0.3f;  // meters (tune this)

    int N = currPoints.size();
    std::vector<bool> visited(N, false);

    for(int i = 0; i < N; i++)
    {
        if(visited[i]) continue;

        // skip invalid points (NaN check)
        if(std::isnan(currPoints[i].x) || std::isnan(prevPoints[i].x))
            continue;

        std::vector<int> cluster;
        cluster.push_back(i);
        visited[i] = true;

        // compute motion of seed point
        cv::Point3f flow_i = currPoints[i] - prevPoints[i];

        for(int j = i + 1; j < N; j++)
        {
            if(visited[j]) continue;

            if(std::isnan(currPoints[j].x) || std::isnan(prevPoints[j].x))
                continue;

            // spatial distance
            float spatial_dist = cv::norm(currPoints[i] - currPoints[j]);

            if(spatial_dist > DIST_THRESH)
                continue;

            // motion consistency
            cv::Point3f flow_j = currPoints[j] - prevPoints[j];
            float motion_dist = cv::norm(flow_i - flow_j);

            if(motion_dist < MOTION_THRESH)
            {
                cluster.push_back(j);
                visited[j] = true;
            }
        }

        // keep only meaningful clusters
        if(cluster.size() >= 5)
            clusters.push_back(cluster);
    }
}

void DynamicTracker::ClusterPoints(
    const std::vector<cv::Point3f>& currPoints,
    const std::vector<cv::Point3f>& prevPoints,
    const GridType& dynamicGrid,
    const std::vector<std::pair<int,int>>& gridPos,
    std::vector<std::vector<int>>& clusters)
{
    const float DIST_THRESH   = 0.5f;
    const float MOTION_THRESH = 0.3f;

    int N = currPoints.size();
    std::vector<bool> visited(N, false);

    for(int i = 0; i < N; i++)
    {
        if(visited[i]) continue;

        if(std::isnan(currPoints[i].x) || std::isnan(prevPoints[i].x))
            continue;

        std::vector<int> cluster;
        std::queue<int> q;

        q.push(i);
        visited[i] = true;

        // Queue for FIFO 
        while(!q.empty())
        {
            int idx = q.front();
            q.pop();

            cluster.push_back(idx);
            if(prevPoints.empty()) return;

            // if(idx>=prevPoints.size()) 
            //     continue;
            //std::cout << "idx: " << idx << std::endl;
            cv::Point3f flow_i = currPoints[idx] - prevPoints[idx];
            float flow_mag = cv::norm(flow_i);

            if(flow_mag > 2.0f)   // tune this
                continue;

            // Get correct grid cell of THIS point
            int gx = gridPos[idx].first;
            int gy = gridPos[idx].second;
            // std::cout << "gx: " << gx << "   gy: " << gy << std::endl;
            // Explore neighboring cells (3x3 cells around the center)
            for(int dx = -1; dx <= 1; dx++)
            {
                for(int dy = -1; dy <= 1; dy++)
                {
                    int nx = gx + dx;
                    int ny = gy + dy;

                    if(nx < 0 || ny < 0 || 
                       nx >= FRAME_GRID_COLS || ny >= FRAME_GRID_ROWS)
                        continue;

                    const auto& cell = dynamicGrid[nx][ny];
                    
                    // Index of keypoints in cell (NOT ITERATING IDX)
                    for(size_t j : cell)
                    {
                        if(j >= currPoints.size() || j >= prevPoints.size())
                        {
                            // std::cout << "INVALID INDEX: " << j << std::endl;
                            continue;
                        }
                        if(visited[j]) continue;

                        if(std::isnan(currPoints[j].x) || std::isnan(prevPoints[j].x))
                            continue;
                        
                        // Check distance between points in real life
                        float spatial_dist = cv::norm(currPoints[idx] - currPoints[j]);
                        if(spatial_dist > DIST_THRESH)
                            continue;
                        
                        // Check if current neighbouring point is moving together with reference point
                        cv::Point3f flow_j = currPoints[j] - prevPoints[j];
                        float motion_dist = cv::norm(flow_i - flow_j);

                        if(motion_dist < MOTION_THRESH)
                        {
                            visited[j] = true;
                            q.push(j);
                        }
                    }
                }
            }
        }
        
        // cluster consists of vector<int> of indices of keypoints detected in Current Frame
        if(cluster.size() >= 5)
            clusters.push_back(cluster);
    }
}

}