#include <queue>
#include <iostream>
#include <iomanip>
#include "FrameDrawer.h"
#include "DynamicTracker.h"
#include "DynamicObject.h"
#include "Detection.h"
#include "Frame.h"
#include "Hungarian.h"
#include <utility>
#include <unordered_map>
#include <limits>

namespace ORB_SLAM3
{
using GridType = std::vector<std::size_t>[FRAME_GRID_COLS][FRAME_GRID_ROWS];

DynamicTracker::DynamicTracker() : next_id(0) {}

float DynamicTracker::ComputeIoU(const cv::Rect& a, const cv::Rect& b)
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

float DynamicTracker::computeSigma(const std::vector<float>& data)
{
    if(data.empty()) return 0.0f;

    float mean = 0.0f;
    for(float x : data) mean += x;
    mean /= (float)data.size();

    float var = 0.0f;
    for(float x : data) var += (x - mean) * (x - mean);
    var /= (float)data.size();

    return std::sqrt(var);
}

void DynamicTracker::rstVars()
{
    // reserved
}

void DynamicTracker::Reset()
{
    next_id = 0;
    onInitialization = true;
    onTrackingLost = false;
}

void DynamicTracker::ProcessFrame(Frame& mCurrentFrame, Frame& mLastFrame,
    const std::vector<std::pair<int,int>>& _idx_matches,
    const std::vector<std::pair<cv::Point3f, cv::Point3f>>& _optical_flow_matches)
{
    if(onInitialization)
    {
        mLastFrame.mDynamicObjects.clear();
        mCurrentFrame.mDynamicObjects.clear();
    }

    if(mCurrentFrame.mImGray.empty()) return;
    if(mLastFrame.mImGrayLast.empty()) return;

    std::vector<std::vector<int>> clusters;
    std::vector<DynamicObject, Eigen::aligned_allocator<DynamicObject>> CurrentObjects;
    std::vector<DynamicObject, Eigen::aligned_allocator<DynamicObject>> PrevObjects;

    clusters = ClusterPoints(mCurrentFrame, mLastFrame);

    cv::Mat frame;
    cv::cvtColor(mCurrentFrame.mImGray.clone(), frame, cv::COLOR_GRAY2BGR);

    // ----------------------------------------------------------------
    // BUILD CURRENT OBJECTS FROM CLUSTERS
    // ----------------------------------------------------------------
    for(const auto& cluster : clusters)
    {
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

        float minx = 1e9f, miny = 1e9f;
        float maxx = -1e9f, maxy = -1e9f;
        int dynamic_pixels = 0;

        for(const auto& kp : objCurr2D)
        {
            int x = (int)kp.pt.x;
            int y = (int)kp.pt.y;

            if(mCurrentFrame.mDynamicMask.at<uchar>(y, x) == 0)
                dynamic_pixels++;

            minx = std::min(minx, kp.pt.x);
            miny = std::min(miny, kp.pt.y);
            maxx = std::max(maxx, kp.pt.x);
            maxy = std::max(maxy, kp.pt.y);
        }

        float ratio = (float)dynamic_pixels / (float)objCurr2D.size();
        if(ratio < 0.5f)
            continue;

        cv::Rect clusterBox(
            cv::Point2f(minx, miny),
            cv::Point2f(maxx, maxy)
        );

        float bestIoU = 0.0f;
        Detection bestDet;

        for(const auto& det : mCurrentFrame.mDetections)
        {
            float iou = ComputeIoU(clusterBox, det.bbox);
            if(iou > bestIoU)
            {
                bestIoU = iou;
                bestDet = det;
            }
        }

        obj.bbox = (bestIoU > 0.7f) ? bestDet.bbox : clusterBox;
        CurrentObjects.push_back(obj);
    }

    // ----------------------------------------------------------------
    // MERGE SPLIT CLUSTERS BELONGING TO SAME YOLO DETECTION
    // ----------------------------------------------------------------
    for(const auto& det : mCurrentFrame.mDetections)
    {
        std::vector<int> insideDet;
        for(int i = 0; i < (int)CurrentObjects.size(); i++)
        {
            float iou = ComputeIoU(CurrentObjects[i].bbox, det.bbox);
            if(iou > 0.2f)
                insideDet.push_back(i);
        }

        if(insideDet.size() <= 1) continue;

        DynamicObject& base = CurrentObjects[insideDet[0]];
        for(int k = 1; k < (int)insideDet.size(); k++)
        {
            int idx = insideDet[k];
            base.points3D.insert(base.points3D.end(),
                CurrentObjects[idx].points3D.begin(),
                CurrentObjects[idx].points3D.end());
            base.points2D.insert(base.points2D.end(),
                CurrentObjects[idx].points2D.begin(),
                CurrentObjects[idx].points2D.end());
        }

        base.ComputeCentroid();
        base.FitEllipsoid();
        base.bbox = det.bbox;

        for(int k = (int)insideDet.size()-1; k >= 1; k--)
            CurrentObjects.erase(CurrentObjects.begin() + insideDet[k]);
    }

    // ----------------------------------------------------------------
    // DENSE DEPTH SAMPLING — enrich pointsHistoryBuffer for stable PCA
    // Called after bbox is finalised; uses depth image stored in Frame.
    // ----------------------------------------------------------------
    if(!mCurrentFrame.mImDepth.empty())
    {
        for(auto& obj : CurrentObjects)
        {
            obj.SampleDepthPoints(
                mCurrentFrame.mImDepth,
                mCurrentFrame.GetPose(),
                Frame::fx, Frame::fy, Frame::cx, Frame::cy,
                mCurrentFrame.mThDepth
            );
        }
    }

    // ----------------------------------------------------------------
    PrevObjects = mLastFrame.mDynamicObjects;

    // ----------------------------------------------------------------
    // COMPUTE PAIRWISE METRICS FOR SIGMA ESTIMATION
    // ----------------------------------------------------------------
    std::vector<float> dist_vect, motion_vect, size_vect;

    for(int i = 0; i < (int)PrevObjects.size(); i++)
    {
        for(int j = 0; j < (int)CurrentObjects.size(); j++)
        {
            cv::Point3f predicted = (PrevObjects[i].kf_initialized)
                ? cv::Point3f(PrevObjects[i].kf_x(0) + PrevObjects[i].kf_x(3),
                              PrevObjects[i].kf_x(1) + PrevObjects[i].kf_x(4),
                              PrevObjects[i].kf_x(2) + PrevObjects[i].kf_x(5))
                : PrevObjects[i].centroid3D;

            float dist     = cv::norm(CurrentObjects[j].centroid3D - predicted);
            float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

            dist_vect.push_back(dist);
            size_vect.push_back(sizeDiff);

            if(PrevObjects[i].velocity.x != -1000.0f)
            {
                cv::Point3f vel_predicted = PrevObjects[i].centroid3D + PrevObjects[i].velocity;
                float motion = cv::norm(CurrentObjects[j].centroid3D - vel_predicted);
                motion_vect.push_back(motion);
            }
        }
    }

    // ----------------------------------------------------------------
    // ASSIGNMENT
    // ----------------------------------------------------------------
    if(CurrentObjects.size() > 0)
    {
        if(PrevObjects.size() > 0)
        {
            // [1,1] — both prev and current exist, run Hungarian
            float sigma_d = std::max(computeSigma(dist_vect),   1e-3f);
            float sigma_m = std::max(computeSigma(motion_vect), 1e-3f);
            float sigma_s = std::max(computeSigma(size_vect),   1e-3f);

            int N = (int)PrevObjects.size();
            int M = (int)CurrentObjects.size();

            std::vector<std::vector<float>> costMatrix(N, std::vector<float>(M, 1e6f));

            for(int i = 0; i < N; i++)
            {
                for(int j = 0; j < M; j++)
                {
                    cv::Point3f kf_predicted = (PrevObjects[i].kf_initialized)
                        ? cv::Point3f(PrevObjects[i].kf_x(0) + PrevObjects[i].kf_x(3),
                                      PrevObjects[i].kf_x(1) + PrevObjects[i].kf_x(4),
                                      PrevObjects[i].kf_x(2) + PrevObjects[i].kf_x(5))
                        : PrevObjects[i].centroid3D;

                    float dist     = cv::norm(CurrentObjects[j].centroid3D - kf_predicted);
                    float sizeDiff = cv::norm(PrevObjects[i].axes - CurrentObjects[j].axes);

                    float d_score = (dist * dist) / (sigma_d * sigma_d);
                    float s_score = (sizeDiff * sizeDiff) / (sigma_s * sigma_s);

                    float iou       = ComputeIoU(PrevObjects[i].bbox, CurrentObjects[j].bbox);
                    float iou_score = 1.0f - iou;

                    float m_score       = 0.0f;
                    float motion_weight = 0.0f;

                    if(PrevObjects[i].velocity.x != -1000.0f)
                    {
                        cv::Point3f vel_predicted = PrevObjects[i].centroid3D + PrevObjects[i].velocity;
                        float motion  = cv::norm(CurrentObjects[j].centroid3D - vel_predicted);
                        m_score       = (motion * motion) / (sigma_m * sigma_m);
                        motion_weight = 0.5f;
                    }

                    float totalScore =
                        1.0f          * d_score +
                        motion_weight * m_score +
                        0.2f          * s_score +
                        1.0f          * iou_score;

                    if(totalScore < 50.0f)
                        costMatrix[i][j] = totalScore;
                }
            }

            std::vector<int> assignment = hungarian_solver.solve(costMatrix);
            std::vector<bool> used(M, false);
            std::vector<DynamicObject, Eigen::aligned_allocator<DynamicObject>> alignedObjects;

            for(int it = 0; it < (int)assignment.size(); it++)
            {
                int j = assignment[it];

                if(j == -1)
                {
                    DynamicObject obj = PrevObjects[it];
                    obj.missed_frames++;

                    if(obj.missed_frames > MAX_MISSED_FRAMES)
                        continue;

                    if(obj.kf_initialized)
                    {
                        Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
                        F(0,3) = 1; F(1,4) = 1; F(2,5) = 1;

                        Eigen::Matrix<float,6,6> Q = Eigen::Matrix<float,6,6>::Zero();
                        Q(0,0) = 0.001f; Q(1,1) = 0.001f; Q(2,2) = 0.001f;
                        Q(3,3) = 0.05f;  Q(4,4) = 0.05f;  Q(5,5) = 0.05f;

                        obj.kf_x = F * obj.kf_x;
                        obj.kf_P = F * obj.kf_P * F.transpose() + Q;

                        obj.centroid3D.x = obj.kf_x(0);
                        obj.centroid3D.y = obj.kf_x(1);
                        obj.centroid3D.z = obj.kf_x(2);

                        obj.velocity.x = obj.kf_x(3);
                        obj.velocity.y = obj.kf_x(4);
                        obj.velocity.z = obj.kf_x(5);
                    }
                    else
                    {
                        obj.centroid3D += obj.velocity;
                    }

                    obj.UpdatePoseFromState();
                    alignedObjects.push_back(obj);
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

                    obj = prev;
                    obj.UpdateFromMeasurement(meas, prev);
                    obj.id = prev.id;
                }

                alignedObjects.push_back(obj);
                used[j] = true;
            }

            // Handle unmatched current objects as new
            for(int j = 0; j < M; j++)
            {
                if(!used[j])
                {
                    DynamicObject obj = CurrentObjects[j];
                    obj.id = next_id++;
                    alignedObjects.push_back(obj);
                }
            }

            mCurrentFrame.mDynamicObjects = alignedObjects;

            // Compute local object-frame points
            for(auto& obj : mCurrentFrame.mDynamicObjects)
            {
                obj.points3D_local.clear();
                for(int i = 0; i < (int)obj.points3D.size(); i++)
                {
                    Eigen::Vector3d Pw    = Converter::toVector3d(obj.points3D[i]);
                    Eigen::Vector3d X_obj = obj.T_obj.inverse() * Pw;
                    obj.points3D_local.push_back(X_obj);
                }
            }

            onInitialization = false;
            onTrackingLost   = false;
        }
        else
        {
            // [0,1] — no prev objects, all current are new
            std::vector<DynamicObject, Eigen::aligned_allocator<DynamicObject>> alignedObjects;
            for(int j = 0; j < (int)CurrentObjects.size(); j++)
            {
                DynamicObject obj = CurrentObjects[j];
                obj.id = next_id++;
                alignedObjects.push_back(obj);
            }

            mCurrentFrame.mDynamicObjects = alignedObjects;
            onInitialization = false;
            onTrackingLost   = false;
        }
    }
    else
    {
        // [x,0] — no current objects detected
        if(PrevObjects.size() > 0)
        {
            // Prediction-only mode
            std::vector<DynamicObject, Eigen::aligned_allocator<DynamicObject>> predictedObjects;

            for(auto& prev : PrevObjects)
            {
                DynamicObject obj = prev;
                obj.missed_frames++;

                if(obj.missed_frames > MAX_MISSED_FRAMES)
                    continue;

                if(obj.kf_initialized)
                {
                    Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
                    F(0,3) = 1; F(1,4) = 1; F(2,5) = 1;

                    Eigen::Matrix<float,6,6> Q = Eigen::Matrix<float,6,6>::Zero();
                    Q(0,0) = 0.001f; Q(1,1) = 0.001f; Q(2,2) = 0.001f;
                    Q(3,3) = 0.05f;  Q(4,4) = 0.05f;  Q(5,5) = 0.05f;

                    obj.kf_x = F * obj.kf_x;
                    obj.kf_P = F * obj.kf_P * F.transpose() + Q;

                    obj.centroid3D.x = obj.kf_x(0);
                    obj.centroid3D.y = obj.kf_x(1);
                    obj.centroid3D.z = obj.kf_x(2);

                    obj.velocity.x = obj.kf_x(3);
                    obj.velocity.y = obj.kf_x(4);
                    obj.velocity.z = obj.kf_x(5);
                }
                else
                {
                    obj.centroid3D += prev.velocity;
                }

                obj.UpdatePoseFromState();
                predictedObjects.push_back(obj);
            }

            mCurrentFrame.mDynamicObjects = predictedObjects;

            // Draw and send to FrameDrawer — no imshow here
            for(auto& obj : mCurrentFrame.mDynamicObjects)
                obj.DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());

            if(mpFrameDrawer && !frame.empty())
                mpFrameDrawer->SetDynamicFrame(frame);

            return;
        }
        else
        {
            // [0,0] — nothing on either side
            if(!onTrackingLost)
            {
                onInitialization = true;
                next_id = 0;
            }
            else
            {
                onTrackingLost = true;
                std::cout << "Tracking Lost!" << std::endl;
            }
        }
    }

    // ----------------------------------------------------------------
    // DRAW — send to FrameDrawer, never imshow directly
    // ----------------------------------------------------------------
    for(auto& obj : mCurrentFrame.mDynamicObjects)
        obj.DrawEllipsoid2D(frame, mCurrentFrame.mK, mCurrentFrame.GetPose());

    if(mpFrameDrawer && !frame.empty())
        mpFrameDrawer->SetDynamicFrame(frame);

    // Debug: frame-level summary for two-person and motion-blur windows
    {
        double ts = mCurrentFrame.mTimeStamp;
        double t  = ts - 1341846314.157989;
        bool in_window = (t >= 1.8 && t <= 2.5) ||  // two-person region
                         (t >= 6.5 && t <= 7.6);     // motion-blur spike region
        if(in_window)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << "[DYNTRACK t=" << t << "s]"
                      << "  yolo_dets=" << mCurrentFrame.mDetections.size()
                      << "  clusters="  << clusters.size()
                      << "  curr_objs=" << CurrentObjects.size()
                      << "  tracked="   << mCurrentFrame.mDynamicObjects.size()
                      << std::endl;
            for(auto& obj : mCurrentFrame.mDynamicObjects)
                std::cout << "    id=" << obj.id
                          << " missed=" << obj.missed_frames
                          << " tracked_frames=" << obj.tracked_frames
                          << " pts3D=" << obj.points3D.size()
                          << " bbox=(" << obj.bbox.x << "," << obj.bbox.y
                          << " " << obj.bbox.width << "x" << obj.bbox.height << ")"
                          << std::endl;
        }
    }
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

        if(prev_kp_idx < 0 || prev_kp_idx >= (int)mLastFrame.mvDynamicPoints3D.size()) continue;
        if(curr_kp_idx < 0 || curr_kp_idx >= M) continue;
        if(visited[curr_kp_idx]) continue;

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

            int prev_idx = -1;
            for(const auto& m : idx_matches)
            {
                if(m.second == curr_idx) { prev_idx = m.first; break; }
            }
            if(prev_idx < 0) continue;

            cluster.push_back(curr_idx);

            cv::Point3f flow_i =
                mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                mLastFrame.mvDynamicPoints3D[prev_idx];

            float depth_i = std::max(mLastFrame.mvDynamicPoints3D[prev_idx].z, 1e-3f);
            flow_i *= (1.0f / depth_i);

            if(cv::norm(flow_i) > 5.0f) continue;

            int gx = mCurrentFrame.mvDynamicGridPos[curr_idx].first;
            int gy = mCurrentFrame.mvDynamicGridPos[curr_idx].second;
            if(gx < 0 || gy < 0) continue;

            const int cell_offset = 2;

            for(int dx = -cell_offset; dx <= cell_offset; dx++)
            {
                for(int dy = -cell_offset; dy <= cell_offset; dy++)
                {
                    int nx = gx + dx;
                    int ny = gy + dy;

                    if(nx < 0 || ny < 0 || nx >= FRAME_GRID_COLS || ny >= FRAME_GRID_ROWS)
                        continue;

                    const std::vector<int>& cell = mCurrentFrame.mDynamicGrid[nx][ny];

                    for(int j = 0; j < (int)cell.size(); j++)
                    {
                        int j_int = cell[j];
                        if(j_int < 0 || j_int >= M || visited[j_int]) continue;

                        int prev_j = -1;
                        for(const auto& m : idx_matches)
                        {
                            if(m.second == j_int) { prev_j = m.first; break; }
                        }
                        if(prev_j < 0) continue;

                        float spatial_dist = cv::norm(
                            mCurrentFrame.mvDynamicPoints3D[curr_idx] -
                            mCurrentFrame.mvDynamicPoints3D[j_int]
                        );
                        if(spatial_dist > DIST_THRESH) continue;

                        cv::Point3f flow_j =
                            mCurrentFrame.mvDynamicPoints3D[j_int] -
                            mLastFrame.mvDynamicPoints3D[prev_j];

                        float depth_j = std::max(mLastFrame.mvDynamicPoints3D[prev_j].z, 1e-3f);
                        flow_j *= (1.0f / depth_j);

                        if(cv::norm(flow_i - flow_j) < MOTION_THRESH)
                        {
                            visited[j_int] = true;
                            q.push(j_int);
                        }
                    }
                }
            }
        }

        if((int)cluster.size() >= 10)
            clusters.push_back(cluster);
    }

    return clusters;
}

std::vector<std::vector<int>> DynamicTracker::ClusterPoints(
    Frame& mCurrentFrame,
    Frame& mLastFrame
)
{
    struct DynamicTrack
    {
        int idx;
        cv::Point2f prev2D, curr2D;
        cv::Point3f prev3D, curr3D, flow3D;
    };

    std::vector<std::vector<int>> clusters;

    if(mLastFrame.mImGrayLast.empty())      return clusters;
    if(mLastFrame.mvDynamicKeys.empty())    return clusters;
    if(mCurrentFrame.mvDynamicKeys.empty()) return clusters;

    std::vector<cv::Point2f> prevPts;
    prevPts.reserve(mLastFrame.mvDynamicKeys.size());

    int N_prev = std::min(mLastFrame.N_dynamic, (int)mLastFrame.mvDynamicKeys.size());
    for(int i = 0; i < N_prev; i++)
        prevPts.push_back(mLastFrame.mvDynamicKeys[i].pt);

    std::vector<cv::Point2f> currPts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        mLastFrame.mImGrayLast,
        mCurrentFrame.mImGray,
        prevPts, currPts, status, err
    );

    std::vector<DynamicTrack> tracks;

    auto validPoint3D = [](const cv::Point3f& p)
    {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };

    for(int k = 0; k < N_prev; k++)
    {
        if(k >= (int)status.size() || !status[k]) continue;
        if(k >= (int)mLastFrame.mvDynamicPoints3D.size()) continue;

        int idx_curr = -1;
        float bestDist = 5.0f;

        for(int i = 0; i < mCurrentFrame.N_dynamic; i++)
        {
            if(i >= (int)mCurrentFrame.mvDynamicKeys.size()) continue;
            float dist = cv::norm(mCurrentFrame.mvDynamicKeys[i].pt - currPts[k]);
            if(dist < bestDist) { bestDist = dist; idx_curr = i; }
        }

        if(idx_curr < 0) continue;
        if(idx_curr >= (int)mCurrentFrame.mvDynamicPoints3D.size()) continue;

        cv::Point3f prev3D = mLastFrame.mvDynamicPoints3D[k];
        cv::Point3f curr3D = mCurrentFrame.mvDynamicPoints3D[idx_curr];

        if(!validPoint3D(prev3D) || !validPoint3D(curr3D)) continue;

        DynamicTrack tr;
        tr.idx    = idx_curr;
        tr.prev2D = prevPts[k];
        tr.curr2D = currPts[k];
        tr.prev3D = prev3D;
        tr.curr3D = curr3D;
        // In ClusterPoints, after computing flow3D for each track,
        // subtract the expected background flow using camera motion
        // Egomotion Compensation 
        Sophus::SE3f Trel = mCurrentFrame.GetPose() * mLastFrame.GetPose().inverse();
        Eigen::Vector3f t_rel = Trel.translation();
        Eigen::Matrix3f R_rel = Trel.rotationMatrix();

        // For each track, compute expected flow if point were static
        //cv::Point3f prev3D = tr.prev3D;
        Eigen::Vector3f Pw(tr.prev3D.x, tr.prev3D.y, tr.prev3D.z);
        Eigen::Vector3f Pc_curr = R_rel * Pw + t_rel;

        cv::Point3f expected_curr(Pc_curr.x(), Pc_curr.y(), Pc_curr.z());
        cv::Point3f ego_flow = expected_curr - prev3D;

        // Subtract ego motion from observed flow
        tr.flow3D = (tr.curr3D - tr.prev3D) - ego_flow;
        //tr.flow3D = curr3D - prev3D;
        tracks.push_back(tr);
    }

    if(tracks.empty()) return clusters;

    const float SPATIAL_THRESH = 0.6f;
    const float MOTION_THRESH  = 0.4f;
    const int   MIN_CLUSTER    = 3;

    std::vector<bool> visited(tracks.size(), false);

    for(size_t i = 0; i < tracks.size(); i++)
    {
        if(visited[i]) continue;

        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        std::vector<int> cluster;

        while(!q.empty())
        {
            int a = q.front();
            q.pop();

            cluster.push_back(tracks[a].idx);

            for(size_t b = 0; b < tracks.size(); b++)
            {
                if(visited[b]) continue;

                float spatialDist = cv::norm(tracks[a].curr3D - tracks[b].curr3D);
                if(spatialDist > SPATIAL_THRESH) continue;

                cv::Point3f flowA = tracks[a].flow3D;
                cv::Point3f flowB = tracks[b].flow3D;

                flowA *= (1.0f / std::max(tracks[a].prev3D.z, 1e-3f));
                flowB *= (1.0f / std::max(tracks[b].prev3D.z, 1e-3f));

                if(cv::norm(flowA - flowB) < MOTION_THRESH)
                {
                    visited[b] = true;
                    q.push(b);
                }
            }
        }

        if((int)cluster.size() >= MIN_CLUSTER)
            clusters.push_back(cluster);
    }

    auto computeBox = [&](const std::vector<int>& cluster) -> cv::Rect2f
    {
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        for(int idx : cluster)
        {
            float x = mCurrentFrame.mvDynamicKeys[idx].pt.x;
            float y = mCurrentFrame.mvDynamicKeys[idx].pt.y;
            minx = std::min(minx, x); miny = std::min(miny, y);
            maxx = std::max(maxx, x); maxy = std::max(maxy, y);
        }
        return cv::Rect2f(minx, miny, maxx - minx, maxy - miny);
    };

    auto expandBox = [](const cv::Rect2f& r, float margin) -> cv::Rect2f
    {
        return cv::Rect2f(r.x - margin, r.y - margin,
                          r.width + 2*margin, r.height + 2*margin);
    };

    bool merged = true;
    while(merged)
    {
        merged = false;
        for(int i = 0; i < (int)clusters.size() && !merged; i++)
        {
            for(int j = i+1; j < (int)clusters.size() && !merged; j++)
            {
                cv::Rect2f bi = expandBox(computeBox(clusters[i]), 30.0f);
                cv::Rect2f bj = computeBox(clusters[j]);

                if((bi & bj).area() > 0)
                {
                    clusters[i].insert(clusters[i].end(),
                        clusters[j].begin(), clusters[j].end());
                    clusters.erase(clusters.begin() + j);
                    merged = true;
                }
            }
        }
    }

    return clusters;
}

} // namespace ORB_SLAM3