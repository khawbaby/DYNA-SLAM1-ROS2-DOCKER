#include <queue>
#include <iostream>
#include <iomanip>
#include "DynamicTracker.h"
#include "Frame.h"
#include "Hungarian.h"
#include <utility>
#include <unordered_map>
#include <limits>
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


void DynamicTracker::rstVars(){
    //dynamic_info = DynamicPtsInfo();
}

void DynamicTracker::ProcessFrame(Frame& mCurrentFrame,Frame& mLastFrame,
    const std::vector<std::pair<int,int>>& _idx_matches,
    const std::vector<std::pair<cv::Point3f, cv::Point3f>>& _optical_flow_matches)
{
    //std::cout << "NEW FRAME" << std::endl;
    optical_flow_matches = _optical_flow_matches;
    idx_matches = _idx_matches;
    // Step 1: cluster indices
    std::vector<std::vector<int>> clusters;
    clusters = ClusterPoints2(mCurrentFrame, mLastFrame);
    std::cout << "Number of Current Objects created after ret: " << clusters.size() << std::endl;
}

void DynamicTracker::ProcessFrame(Frame& mCurrentFrame,Frame& mLastFrame, 
    const std::vector<cv::Point3f>& _currPoints,
    const std::vector<cv::Point3f>& _prevPoints, 
    const std::vector<std::pair<int,int>>& _dynamicMatchesIndex)
{
    
    // std::cout << "curr size: " << _currPoints.size() << std::endl;
    // std::cout << "prev size: " << _prevPoints.size() << std::endl;
    // std::cout << "matches size: " << _dynamicMatchesIndex.size() << std::endl;

    dynamic_info.currPoints3D = _currPoints;
    dynamic_info.prevPoints3D = _prevPoints;
    dynamic_info.dynamicMatchesIndex = _dynamicMatchesIndex;

    // Step 1: cluster indices
    std::vector<std::vector<int>> clusters;
    ClusterPoints(mCurrentFrame, mCurrentFrame.mGrid, mCurrentFrame.mvGridPos, clusters);
    //std::cout << "Number of Current Objects created: " << clusters.size() << std::endl;
    
    // Clusters consists of vector of points
    // for(const auto& cluster : clusters)
    // {
    //     std::vector<cv::Point3f> objCurr;
    //     std::vector<cv::Point3f> objPrev;

    //     for(int idx : cluster)
    //     {
    //         objCurr.push_back(mCurrentFrame.mvPoints3D[idx]);
    //         objPrev.push_back(mLastFrame.mvPoints3D[idx]);
    //     }

    //     DynamicObject obj(next_id++);
    //     obj.Update(objCurr, objPrev);
    //     obj.ComputeCentroid();
    //     obj.FitEllipsoid();
    //     CurrentObjects.push_back(obj);
    // }


    // // Compute for all pairs
    // std::vector<float> dist_vect, motion_vect, size_vect;
    // for(int i = 0; i < PrevObjects.size(); i++)
    // {
    //     for(int j = 0; j < CurrentObjects.size(); j++)
    //     {
    //         float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);

    //         cv::Point3f vel_old = PrevObjects[i].centroid3D - PrevObjects[i].prevCentroid3D;
    //         cv::Point3f predicted = PrevObjects[i].centroid3D + vel_old;

    //         float motion = cv::norm(CurrentObjects[j].centroid3D - predicted);

    //         float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

    //         dist_vect.push_back(dist);
    //         motion_vect.push_back(motion);
    //         size_vect.push_back(sizeDiff);

    //         // std::cout
    //         //     << " dist: " << dist << std::setw(25)
    //         //     << " motion: " << motion << std::setw(25)
    //         //     << " sizeDiff: " << sizeDiff << std::endl;
    //         // std::cout << std::endl;
    //     }
    // }

    // // Compute sigma
    // float sigma_d = computeSigma(dist_vect);
    // float sigma_m = computeSigma(motion_vect);
    // float sigma_s = computeSigma(size_vect);
    
    // sigma_d = std::max(sigma_d, 1e-3f);
    // sigma_m = std::max(sigma_m, 1e-3f);
    // sigma_s = std::max(sigma_s, 1e-3f);

    // int N = PrevObjects.size();
    // int M = CurrentObjects.size();

    // std::vector<std::vector<float>> costMatrix(N, std::vector<float>(M, 1e6));

    // // Fill up Cost Matrix
    // for(int i = 0; i < N; i++)
    // {
    //     for(int j = 0; j < M; j++)
    //     {
    //         float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);

    //         cv::Point3f vel_old = PrevObjects[i].centroid3D - PrevObjects[i].prevCentroid3D;
    //         cv::Point3f predicted = PrevObjects[i].centroid3D + vel_old;

    //         float motion = cv::norm(CurrentObjects[j].centroid3D - predicted);

    //         float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

    //         float d_score = (dist*dist)/(sigma_d*sigma_d);
    //         float m_score = (motion*motion)/(sigma_m*sigma_m);
    //         float s_score = (sizeDiff*sizeDiff)/(sigma_s*sigma_s);

    //         float totalScore = d_score + m_score + s_score;
    //         // std::cout
    //         //     << " dist: " << d_score << std::setw(25)
    //         //     << " motion: " << m_score << std::setw(25)
    //         //     << " sizeDiff: " << s_score << std::endl;
    //         // std::cout << std::endl;

    //         // Gating
    //         if(totalScore < 50.0f)
    //             costMatrix[i][j] = totalScore;
    //     }
    // }
    
    // //std::vector<int> assignment = hungarian_solver.solve(costMatrix);

    // std::vector<bool> usedDetections(M, false);
    
}

float DynamicTracker::Distance(const cv::Point3f& a, const cv::Point3f& b)
{
    return cv::norm(a - b);
}

std::vector<std::vector<int>> DynamicTracker::ClusterPoints2(
    Frame& mCurrentFrame,
    Frame& mLastFrame
)
{

    int N = optical_flow_matches.size();
    int M = mCurrentFrame.N_dynamic;
    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(M, false);

    for(int i = 0; i < N; i++)
    {
        // where is this in mvDynamicPoints3D, index for points
        int prev_kp_idx = idx_matches[i].first;
        int curr_kp_idx = idx_matches[i].second;
       
        if(visited[curr_kp_idx]) continue;

        if(std::isnan(mCurrentFrame.mvDynamicPoints3D[curr_kp_idx].x) || std::isnan(mLastFrame.mvDynamicPoints3D[prev_kp_idx].x))
            continue;

        std::vector<int> cluster;
        std::queue<int> q;

        q.push(curr_kp_idx);
        visited[curr_kp_idx] = true;

        // Queue for FIFO 
        while(!q.empty())
        {
            // Start as reference point 
            int curr_idx = q.front();
            int prev_idx;
            for(const auto& m : idx_matches)
            {
                if(m.second == curr_idx)
                {
                    prev_idx = m.first;
                    break;
                }
            }
            q.pop();

            cluster.push_back(curr_idx);  // store mvKeys index
        
            // Reference flow
            cv::Point3f flow_i = mCurrentFrame.mvDynamicPoints3D[curr_idx] - mLastFrame.mvDynamicPoints3D[prev_idx];
            float flow_mag = cv::norm(flow_i);
            
            if(flow_mag > 2.0f)   // tune this
                continue;

            //std::cout << "flow_i:\t" << flow_i << std::endl;

            // Get correct grid cell of THIS point
            int gx = mCurrentFrame.mvDynamicGridPos[curr_idx].first;
            int gy = mCurrentFrame.mvDynamicGridPos[curr_idx].second;
            
            if (gx < 0 || gy < 0) continue;

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

                    const auto& cell = mCurrentFrame.mDynamicGrid[nx][ny];
                    
                    // Index of keypoints in cell (NOT ITERATING IDX)
                    for(size_t j : cell)
                    { 
                        int j_int ;
                        if(j <= std::numeric_limits<int>::max())
                        {
                            j_int = static_cast<int>(j);
                        }
                        else
                        {
                            continue;
                        }
                        if(visited[j_int]) continue;

                        // Check if current point has prev correspondence
                        // If no prev found, discard
                        bool found = false;
                        cv::Point3f Potential_Pt_prev;
                        int prev_j;
                        // Find prev for potential point
                        for(const auto& m : idx_matches)
                        {
                            if(m.second == j_int)
                            {
                                prev_j = m.first;
                                // std::cout 
                                // << "Curr Idx: \t" << m.second 
                                // << "\tj_int: \t" << j_int 
                                // << "\t prev_j: \t" << prev_j 
                                // << std::endl;
                                found = true;
                                break;
                            }
                        }
                        
                        if (!found) continue;

                        // Dist between reference and potential 
                        float spatial_dist = cv::norm(
                            mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                            mCurrentFrame.mvDynamicPoints3D[j_int]
                        );

                        if(spatial_dist > DIST_THRESH) continue;
                        
                        // Flow between curr and prev (potential point)
                        cv::Point3f flow_j =
                            mCurrentFrame.mvDynamicPoints3D[j_int] -
                            mLastFrame.mvDynamicPoints3D[prev_j];
                        
                        //std::cout << "flow_j:\t" << flow_j << std::endl;
                        
                        float motion_dist = cv::norm(flow_i - flow_j);
                        
                        //std::cout << "Motion:\t" << motion_dist << "\n" << std::endl;

                        if(motion_dist < MOTION_THRESH)
                        {
                            visited[j_int] = true;
                            q.push(j_int);
                        }
                    }
                }
            }
        }
        
        // cluster consists of vector<int> of indices of keypoints detected in Current Frame
        if(cluster.size() >= 5)
            clusters.push_back(cluster); 
            //std::cout << "New Objects Created !" << std::endl;
    }

    std::cout << "Number of Objects Created : " << clusters.size() << std::endl;
    return clusters;
}

void DynamicTracker::ClusterPoints(
    Frame& mCurrentFrame,
    const GridType& Grid,
    const std::vector<std::pair<int,int>>& GridPos,
    std::vector<std::vector<int>>& clusters)
    {
        const float DIST_THRESH   = 0.5f;
        const float MOTION_THRESH = 0.3f;

        int N = GridPos.size();
        std::vector<bool> visited(N, false);

        // Map to a pair of <kp_idx, filtered_idx>
        std::unordered_map<int,int> currToMatchIdx;
        //std::cout << "Matches" << dynamic_info.dynamicMatchesIndex.size() << std::endl;
        for(int i = 0; i < dynamic_info.dynamicMatchesIndex.size(); i++)
        {
            auto it_curr = dynamic_info.dynamicMatchesIndex[i];
            currToMatchIdx[it_curr.second] = i;
        }

        // i is index of filtered subset
        for(int i = 0; i < dynamic_info.dynamicMatchesIndex.size(); i++)
        {
            // Map MatchIndex[i] to index of mvKeys (Take current)
            auto it_kp_idx = dynamic_info.dynamicMatchesIndex[i];
            int kp_idx = it_kp_idx.second;
            if(visited[kp_idx]) continue;

            if(std::isnan(dynamic_info.currPoints3D[i].x) || std::isnan(dynamic_info.prevPoints3D[i].x))
                continue;

            std::vector<int> cluster;
            std::queue<int> q;

            q.push(kp_idx);
            visited[kp_idx] = true;

            // Queue for FIFO 
            while(!q.empty())
            {
                // idx is index of mvKeys
                int kp = q.front();
                q.pop();

                cluster.push_back(kp);  // store mvKeys index

                // dyn_i, to compare reference point to all neighbouring points
                auto it_i = currToMatchIdx.find(kp);
                if(it_i == currToMatchIdx.end()) continue;
                int dyn_i = it_i->second;

                if(dynamic_info.prevPoints3D.empty()) return;

                // if(idx>=prevPoints.size()) 
                //     continue;
                //std::cout << "idx: " << idx << std::endl;
                cv::Point3f flow_i = dynamic_info.currPoints3D[dyn_i] - dynamic_info.prevPoints3D[dyn_i];
                float flow_mag = cv::norm(flow_i);

                if(flow_mag > 2.0f)   // tune this
                    continue;
                    
                // Get correct grid cell of THIS point
                int gx = GridPos[kp].first;
                int gy = GridPos[kp].second;
     
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

                        const auto& cell = Grid[nx][ny];
                        
                        // Index of keypoints in cell (NOT ITERATING IDX)
                        for(size_t j : cell)
                        {
                            // j is mvKeys index

                            if(!mCurrentFrame.mvbDynamic[j]) continue;
                            if(visited[j]) continue;
                            
                            // dyn_j is for neighbouring points to match to dyn_i
                            // find corresponding dynamic match index
                            auto it_j = currToMatchIdx.find(j);
                            if(it_j == currToMatchIdx.end()) continue;

                            int dyn_j = it_j->second;

                            if(std::isnan(dynamic_info.currPoints3D[dyn_j].x) ||
                            std::isnan(dynamic_info.prevPoints3D[dyn_j].x))
                                continue;

                            float spatial_dist = cv::norm(
                                dynamic_info.currPoints3D[dyn_i] -
                                dynamic_info.currPoints3D[dyn_j]
                            );

                            if(spatial_dist > DIST_THRESH) continue;

                            cv::Point3f flow_j =
                                dynamic_info.currPoints3D[dyn_j] -
                                dynamic_info.prevPoints3D[dyn_j];

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