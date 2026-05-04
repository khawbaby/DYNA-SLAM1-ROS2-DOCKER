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
        // obj.DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());
        CurrentObjects.push_back(obj);
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
    
    // Assignment 
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
                    const DynamicObject& prev = PrevObjects[it];
                    const DynamicObject& meas = CurrentObjects[j];

                    obj = prev;  // keep state
                    obj.UpdateFromMeasurement(meas, prev);
                    obj.id = prev.id;
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
            
            for(auto &obj : mCurrentFrame.mDynamicObjects) {
                // obj.points3D_local.clear();
                for(int i=0; i<obj.points3D.size(); i++) {
                    Eigen::Vector3d Pw = Converter::toVector3d(obj.points3D[i]);
                    Eigen::Vector3d X_obj = obj.T_obj.inverse() * Pw;

                    obj.points3D_local.push_back(X_obj);
                }
            }

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
        // 🔥 PREDICTION-ONLY MODE (no detections this frame)
        // if(CurrentObjects.size() == 0 && PrevObjects.size() > 0)
        // {
        //     std::vector<DynamicObject> predictedObjects;

        //     for(auto& prev : PrevObjects)
        //     {
        //         DynamicObject obj = prev;
        //         obj.missed_frames++;   // 🔥 increment

        //         // Drop if too old
        //         if(obj.missed_frames > MAX_MISSED_FRAMES)
        //             continue;
        //         // --- Kalman predict ONLY ---
        //         if(obj.kf_initialized)
        //         {
        //             Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
        //             F(0,3)=1; F(1,4)=1; F(2,5)=1;

        //             Eigen::Matrix<float,6,6> Q = Eigen::Matrix<float,6,6>::Identity() * 0.01f;

        //             obj.kf_x = F * obj.kf_x;
        //             obj.kf_P = F * obj.kf_P * F.transpose() + Q;

        //             // write back
        //             obj.centroid3D.x = obj.kf_x(0);
        //             obj.centroid3D.y = obj.kf_x(1);
        //             obj.centroid3D.z = obj.kf_x(2);

        //             obj.velocity.x = obj.kf_x(3);
        //             obj.velocity.y = obj.kf_x(4);
        //             obj.velocity.z = obj.kf_x(5);
        //         }
        //         else
        //         {
        //             // fallback (if KF not initialized)
        //             obj.centroid3D += prev.velocity;
        //         }

        //         obj.UpdatePoseFromState();
        //         predictedObjects.push_back(obj);
        //     }

        //     mCurrentFrame.mDynamicObjects = predictedObjects;
        //     for(int j = 0; j < mCurrentFrame.mDynamicObjects.size(); j++)
        //     {
        //         std::cout << mCurrentFrame.mDynamicObjects[j].id << "\t";
        //         mCurrentFrame.mDynamicObjects[j].DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());
        //     }

        //     if(!frame.empty()) {
        //         cv::imshow("Ellipsoids", frame);
        //         cv::waitKey(1);
        //     }
        //     return;  // 🚨 IMPORTANT: skip rest of pipeline
        // }
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
        mCurrentFrame.mDynamicObjects[j].DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());
    }

    if(!frame.empty()) {
        cv::imshow("Ellipsoids", frame);
        cv::waitKey(1);
    }
    
    std::cout << "\nNext ID: " << next_id << std::endl;
    //mCurrentFrame.mDynamicObjects = alignedObjects;
    std::cout << std::endl;
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
        int prev_kp_idx = idx_matches[i].first;
        int curr_kp_idx = idx_matches[i].second;

        if(curr_kp_idx < 0 || curr_kp_idx >= M) continue;
        if(visited[curr_kp_idx]) continue;

        // invalid depth check
        if(std::isnan(mCurrentFrame.mvDynamicPoints3D[curr_kp_idx].z) ||
           std::isnan(mLastFrame.mvDynamicPoints3D[prev_kp_idx].z))
            continue;

        std::vector<int> cluster;
        std::queue<int> q;

        q.push(curr_kp_idx);
        visited[curr_kp_idx] = true;

        while(!q.empty())
        {
            int curr_idx = q.front();
            q.pop();

            // find corresponding prev index
            int prev_idx = -1;
            for(const auto& m : idx_matches)
            {
                if(m.second == curr_idx)
                {
                    prev_idx = m.first;
                    break;
                }
            }
            if(prev_idx < 0) continue;

            cluster.push_back(curr_idx);

            // --- FLOW (DEPTH NORMALIZED) ---
            cv::Point3f flow_i =
                mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                mLastFrame.mvDynamicPoints3D[prev_idx];

            float depth_i = std::max(mLastFrame.mvDynamicPoints3D[prev_idx].z, 1e-3f);
            flow_i *= (1.0f / depth_i);

            float flow_mag = cv::norm(flow_i);

            // relaxed gating (important)
            if(flow_mag > 5.0f)
                continue;

            // --- GRID POSITION ---
            int gx = mCurrentFrame.mvDynamicGridPos[curr_idx].first;
            int gy = mCurrentFrame.mvDynamicGridPos[curr_idx].second;

            if(gx < 0 || gy < 0) continue;

            // smaller search window (important)
            int cell_offset = 2;

            for(int dx = -cell_offset; dx <= cell_offset; dx++)
            {
                for(int dy = -cell_offset; dy <= cell_offset; dy++)
                {
                    int nx = gx + dx;
                    int ny = gy + dy;

                    if(nx < 0 || ny < 0 ||
                       nx >= FRAME_GRID_COLS || ny >= FRAME_GRID_ROWS)
                        continue;

                    const std::vector<int>& cell = mCurrentFrame.mDynamicGrid[nx][ny];

                    for(int j = 0; j < (int)cell.size(); j++)
                    {
                        int j_int = cell[j];

                        if(j_int < 0 || j_int >= M) continue;
                        if(visited[j_int]) continue;

                        // find prev correspondence
                        int prev_j = -1;
                        for(const auto& m : idx_matches)
                        {
                            if(m.second == j_int)
                            {
                                prev_j = m.first;
                                break;
                            }
                        }
                        if(prev_j < 0) continue;

                        // --- SPATIAL CONSISTENCY ---
                        float spatial_dist = cv::norm(
                            mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                            mCurrentFrame.mvDynamicPoints3D[j_int]
                        );

                        if(spatial_dist > DIST_THRESH)
                            continue;

                        // --- FLOW CONSISTENCY (DEPTH NORMALIZED) ---
                        cv::Point3f flow_j =
                            mCurrentFrame.mvDynamicPoints3D[j_int] -
                            mLastFrame.mvDynamicPoints3D[prev_j];

                        float depth_j = std::max(mLastFrame.mvDynamicPoints3D[prev_j].z, 1e-3f);
                        flow_j *= (1.0f / depth_j);

                        float motion_dist = cv::norm(flow_i - flow_j);

                        if(motion_dist < MOTION_THRESH)
                        {
                            visited[j_int] = true;
                            q.push(j_int);
                        }
                    }
                }
            }
        }

        if(cluster.size() >= 10) // minimum cluster size
            clusters.push_back(cluster);
    }

    return clusters;
}

}

