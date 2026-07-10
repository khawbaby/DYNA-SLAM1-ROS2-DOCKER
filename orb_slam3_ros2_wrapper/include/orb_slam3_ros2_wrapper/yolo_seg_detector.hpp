#ifndef ORB_SLAM3_ROS2_WRAPPER_YOLO_SEG_DETECTOR_HPP_
#define ORB_SLAM3_ROS2_WRAPPER_YOLO_SEG_DETECTOR_HPP_

// In-process YOLOv8-seg inference (ONNX Runtime), called synchronously from
// RGBDCallback on the exact frame being tracked. Replaces the old yolo_node.py
// + async mask/detections topic pair, which had no timestamp sync to the
// RGB/D pair and could feed TrackRGBD a mask that was a cycle (or more)
// behind the object's true position - see the dynamic-mask motion
// compensation added to ORB_SLAM3's Tracking.cc/Frame.cc for the workaround
// that was needed before this existed.
//
// Pre/postprocessing validated against ultralytics' own output on the same
// frame before integration (0.971 mask IoU, detections within ~0.5%
// confidence and a few px) - see ORB_SLAM3/cpp_yolo_prototype/.

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

#include "Detection.h"

namespace ORB_SLAM3_Wrapper
{

struct YoloSegResult
{
    cv::Mat mask;  // CV_8UC1, same size as input frame. 1 = static, 0 = dynamic.
    std::vector<ORB_SLAM3::Detection> detections;  // dynamic-class detections only
};

class YoloSegDetector
{
public:
    // modelPath: path to a YOLOv8-seg .onnx export (e.g. yolov8n-seg.onnx).
    // useCuda: request the CUDA execution provider; falls back to CPU if
    // it fails to load (e.g. missing GPU libs) rather than throwing, so a
    // misconfigured environment degrades to slow-but-working instead of dead.
    YoloSegDetector(const std::string& modelPath, bool useCuda);

    // BGR input, any resolution. Synchronous - blocks until done.
    YoloSegResult Run(const cv::Mat& bgrFrame);

private:
    struct Candidate
    {
        float x1, y1, x2, y2;  // original-image pixel coords
        int classId;
        float conf;
        std::vector<float> maskCoeffs;
    };

    struct LetterboxInfo
    {
        float scale;
        int padLeft, padTop;
        int unpadW, unpadH;
    };

    static cv::Mat Letterbox(const cv::Mat& img, LetterboxInfo& info);
    static float IoU(const Candidate& a, const Candidate& b);
    static std::vector<Candidate> NonMaxSuppression(std::vector<Candidate> dets);

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string inputName_;
    std::string output0Name_;
    std::string output1Name_;
};

}  // namespace ORB_SLAM3_Wrapper

#endif  // ORB_SLAM3_ROS2_WRAPPER_YOLO_SEG_DETECTOR_HPP_
