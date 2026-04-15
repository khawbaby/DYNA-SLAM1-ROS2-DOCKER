#include <queue>
#include <iostream>
#include <iomanip>
#include "DynamicTracker.h"
#include "DynamicObject.h"
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
    for(float x : data) {
        if (x<0)
            return x;
        mean += x;
    }
    mean /= data.size();

    // Variance
    float var = 0.0f;
    for(float x : data)
        var += (x - mean) * (x - mean);
    var /= data.size();   // use (size - 1) if you want unbiased

    return std::sqrt(var);
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
    std::vector<DynamicObject> CurrentObjects;
    std::vector<DynamicObject> PrevObjects;
    clusters = ClusterPoints2(mCurrentFrame, mLastFrame);
    // cv::imshow("Ellipsoids", mCurrentFrame.mImGray);
    // cv::waitKey(1);
    cv::Mat frame;
    cv::cvtColor(mCurrentFrame.mImGray.clone(), frame, cv::COLOR_GRAY2BGR);

    for(const auto& cluster : clusters){
        std::vector<cv::Point3f> objCurr;
        std::vector<cv::KeyPoint> objCurr2D;
        for(int idx : cluster)
        {
            objCurr.push_back(mCurrentFrame.mvDynamicPoints3D[idx]);
            objCurr2D.push_back(mCurrentFrame.mvDynamicKeys[idx]);
        }
         
        DynamicObject obj(0);
        obj.Update(objCurr, objCurr2D);
        obj.ComputeCentroid();
        obj.FitEllipsoid();
        obj.DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());
        CurrentObjects.push_back(obj);
    }

    if(!frame.empty()) {
        cv::imshow("Ellipsoids", frame);
        cv::waitKey(1);
    }
    
    mCurrentFrame.mDynamicObjects = CurrentObjects;
    PrevObjects = mLastFrame.mDynamicObjects;

    std::cout << "Prev Objects: " << PrevObjects.size() << std::endl;
    std::cout << "Current Objects: " << CurrentObjects.size() << std::endl;
    // std::cout << "Ret Objects Created : " << clusters.size() << std::endl;
    // std::cout << "Curr Objects Created : " << CurrentObjects.size() << std::endl;
    // std::cout << "Prev Objects Created : " << mLastFrame.mDynamicObjects.size() << "\n" << std::endl;
    bool newObject = false;
    
    // Compute for all pairs
    std::vector<float> dist_vect, motion_vect, size_vect;
    for(int i = 0; i < PrevObjects.size(); i++)
    {
        std::cout << "Velocity: " << PrevObjects[i].velocity << std::endl;
        for(int j = 0; j < CurrentObjects.size(); j++)
        {
            float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);

            float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);
            dist_vect.push_back(dist);
            size_vect.push_back(sizeDiff);
            
            float motion = -1;
            if (PrevObjects[i].velocity.x != -1000) {
                cv::Point3f predicted = PrevObjects[i].centroid3D + PrevObjects[i].velocity;
                motion = cv::norm(CurrentObjects[j].centroid3D - predicted);
                motion_vect.push_back(motion);
            } 
            
            // std::cout
            //     << " dist0: " << dist << std::setw(25)
            //     << " motion0: " << motion << std::setw(25)
            //     << " sizeDiff0: " << sizeDiff << std::endl;
            // std::cout << std::endl;
        }
    }
    
    if (CurrentObjects.size() > 0) {
        // [1,1] or [1,0]
        if (PrevObjects.size() > 0) {
            // [1,1]
            // Compute sigma
            float sigma_d = computeSigma(dist_vect);
            float sigma_m = computeSigma(motion_vect);
            float sigma_s = computeSigma(size_vect);
            
            sigma_d = std::max(sigma_d, 1e-3f);
            sigma_m = std::max(sigma_m, 1e-3f);
            sigma_s = std::max(sigma_s, 1e-3f);

            std::cout
                << " sigma_d: " << sigma_d << std::setw(25)
                << " sigma_m: " << sigma_m << std::setw(25)
                << " sigma_s: " << sigma_s << std::endl;

            int N = PrevObjects.size();
            int M = CurrentObjects.size();

            std::vector<std::vector<float>> costMatrix(N, std::vector<float>(M, 1e6));

            // Fill up Cost Matrix
            for(int i = 0; i < N; i++)
            {
                for(int j = 0; j < M; j++)
                {
                    float dist = cv::norm(CurrentObjects[j].centroid3D - PrevObjects[i].centroid3D);
                    float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);
                    float m_score = -1; 
                    float motion = 0;

                    if (PrevObjects[i].velocity.x != -1000) {
                        cv::Point3f predicted = PrevObjects[i].centroid3D + PrevObjects[i].velocity;
                        motion = cv::norm(CurrentObjects[j].centroid3D - predicted);
                    } 

                    float d_score = (dist*dist)/(sigma_d*sigma_d);
                    float s_score = (sizeDiff*sizeDiff)/(sigma_s*sigma_s);
                    m_score = (motion*motion)/(sigma_m*sigma_m);

                    float totalScore = d_score + m_score + s_score;
                    std::cout
                        << " d_score: " << d_score << std::setw(25)
                        << " m_score: " << m_score << std::setw(25)
                        << " s_score: " << s_score << std::endl;

                    // Gating
                    if(totalScore < 50.0f)
                        costMatrix[i][j] = totalScore;
                }
            } 

            // [A1, A2, A3] -> Prev Obj
            // [1,  0,  2] -> [A1->B2,  A2->B1,  A3->B3]
            // [B1, B2, B3] -> Curr Obj
            
            std::vector<int> assignment = hungarian_solver.solve(costMatrix);
            std::cout << "Assignments: " << assignment.size() << std::endl;
            std::vector<bool> used(CurrentObjects.size(), false);
            std::cout << std::endl;

            std::vector<DynamicObject> alignedObjects; 
            int it=0;
            int idx =0;

            for(int it = 0; it < assignment.size(); it++)
            {
                int j = assignment[it];
                
                if(j == -1) 
                {
                    std::cout << "Tracking lost\n";
                    continue;
                }

                DynamicObject obj = CurrentObjects[j];

                if(onInitialization)
                {
                    obj.id = next_id++;
                }
                else
                {
                    obj.id = PrevObjects[it].id;
                    obj.velocity = obj.centroid3D - PrevObjects[it].centroid3D;
                }

                alignedObjects.push_back(obj);
                used[j] = true;
            }

            // Handle NEW objects
            for(int j = 0; j < CurrentObjects.size(); j++)
            {
                if(!used[j])
                {
                    DynamicObject obj = CurrentObjects[j];
                    obj.id = next_id++;
                    alignedObjects.push_back(obj);
                }
            }

            mCurrentFrame.mDynamicObjects = alignedObjects;
            onInitialization = false;
            onTrackingLost = false;
        } else { // [0,1]
                // HANDLE NEW OBJ
            std::vector<DynamicObject> alignedObjects; 
            for(int j = 0; j < CurrentObjects.size(); j++)
            {
                DynamicObject obj = CurrentObjects[j];
                obj.id = next_id++;
                alignedObjects.push_back(obj);
            }
            mCurrentFrame.mDynamicObjects = alignedObjects;
            onInitialization = false;
            onTrackingLost = false;
        }
    } else { //[1,0]
        if (PrevObjects.size()>0) {
            onTrackingLost = true;
            std::cout << "1->0 condition, Tracking Lost !" << std::endl;
        } else { // [0,0]
            if (!onTrackingLost) {
                onInitialization = true;
                next_id = 0;
            }
        }
        
    }
    
    for(int j = 0; j < mCurrentFrame.mDynamicObjects.size(); j++)
    {
        std::cout << mCurrentFrame.mDynamicObjects[j].id << "\t";
    }

    std::cout << "\nNext ID: " << next_id << std::endl;
    //mCurrentFrame.mDynamicObjects = alignedObjects;
    std::cout << std::endl;
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
    //ClusterPoints(mCurrentFrame, mCurrentFrame.mGrid, mCurrentFrame.mvGridPos, clusters);
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
       
        int idx = 0;
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
            //std::cout << "Cluster Size: " << cluster.size() << std::endl;
            // Reference flow
            cv::Point3f flow_i = mCurrentFrame.mvDynamicPoints3D[curr_idx] - mLastFrame.mvDynamicPoints3D[prev_idx];
            float flow_mag = cv::norm(flow_i);
            
            //std::cout << "flow_mag:\t" << flow_mag << std::endl;
            if(flow_mag > 2.0f)   // tune this
                continue;
            
            int gx, gy;
            // // Get correct grid cell of THIS point
            // std::cout << "gx: " << gx << "\tgy: " << gy << std::endl;
            gx = mCurrentFrame.mvDynamicGridPos[curr_idx].first;
            gy = mCurrentFrame.mvDynamicGridPos[curr_idx].second;
            
            //std::cout << "gx: " << gx << "\tgy: " << gy << std::endl;

            if (gx < 0 || gy < 0) continue;

            int cell_offset_x = 40;
            int cell_offset_y = 40;
            // Explore neighboring cells (3x3 cells around the center)
            for(int dx = -cell_offset_x; dx <= cell_offset_x; dx++)
            {
                for(int dy = -cell_offset_x; dy <= cell_offset_x; dy++)
                {
                    
                    int nx = gx + dx;
                    int ny = gy + dy;
                    
                    if(nx < 0 || ny < 0 || 
                    nx >= FRAME_GRID_COLS || ny >= FRAME_GRID_ROWS)
                        continue;
                    
                    std::vector<int> cell = mCurrentFrame.mDynamicGrid[nx][ny];
                    //std::cout << "nx: " << nx << "\tny: " << ny << "\tcell size: " << cell.size() << std::endl;
                    // Index of keypoints in cell (NOT ITERATING IDX)
                    for(int j=0; j<cell.size(); j++)
                    { 
                        int j_int = cell[j];
                        
                        if(visited[j_int]) continue;
                        
                        // Check if current point has prev correspondence
                        // If no prev found, discard
                        bool found = false;
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
                                // Dist between reference and potential 
                                float spatial_dist = cv::norm(
                                    mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                                    mCurrentFrame.mvDynamicPoints3D[j_int]
                                );
                                    
                                
                                if(spatial_dist > DIST_THRESH) {
                                    //std::cout << "spatial_dist:\t" << spatial_dist << std::endl;
                                    continue;
                                }
                                // Flow between curr and prev (potential point)
                                cv::Point3f flow_j =
                                    mCurrentFrame.mvDynamicPoints3D[j_int] -
                                    mLastFrame.mvDynamicPoints3D[prev_j];
                                
                                //std::cout << "flow_j:\t" << flow_j << std::endl;
                                
                                float motion_dist = cv::norm(flow_i - flow_j);
                                //std::cout << "Motion:\t" << motion_dist << std::endl;

                                if(motion_dist < MOTION_THRESH)
                                {
                                    visited[j_int] = true;
                                    q.push(j_int);
                                }
                            }
                        }
                        
                        if (!found) continue;

                    }
                }
            }
            //std::cout << "Queue Size: " << q.size() << std::endl;
        }
        
        // Queue empty
        // cluster consists of vector<int> of indices of keypoints detected in Current Frame
        if(cluster.size() >= 5) {
            clusters.push_back(cluster); 
            idx++; 
            //std::cout << "Queue Empty and cluster Size > 5, Move on next point \t Index : \t " << idx << std::endl;
        }
    }

    
    return clusters;
}

}