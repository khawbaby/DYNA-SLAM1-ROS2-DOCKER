#include "orb_slam3_ros2_wrapper/yolo_seg_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ORB_SLAM3_Wrapper
{

namespace
{
constexpr int kInputSize = 640;
constexpr int kNumClasses = 80;
constexpr int kNumMaskCoeffs = 32;
constexpr int kProtoSize = 160;
constexpr float kConfThresh = 0.25f;
constexpr float kIouThresh = 0.7f;

// Same dynamic_classes list as the retired yolo_node.py: person, bicycle,
// car, motorcycle, bus, truck.
const std::vector<int> kDynamicClasses = {0, 1, 2, 3, 5, 7};

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

float YoloSegDetector::IoU(const Candidate& a, const Candidate& b)
{
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

YoloSegDetector::YoloSegDetector(const std::string& modelPath, bool useCuda)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yolo_seg_detector");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);

    if (useCuda) {
        try {
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        } catch (const Ort::Exception& e) {
            std::cerr << "[YoloSegDetector] CUDA execution provider unavailable ("
                      << e.what() << "), falling back to CPU\n";
        }
    }

    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);

    auto inputNameAlloc = session_->GetInputNameAllocated(0, allocator_);
    auto output0NameAlloc = session_->GetOutputNameAllocated(0, allocator_);
    auto output1NameAlloc = session_->GetOutputNameAllocated(1, allocator_);
    inputName_ = inputNameAlloc.get();
    output0Name_ = output0NameAlloc.get();
    output1Name_ = output1NameAlloc.get();

    std::cout << "[YoloSegDetector] loaded " << modelPath << " (CUDA "
              << (useCuda ? "requested" : "disabled") << ")\n";
}

cv::Mat YoloSegDetector::Letterbox(const cv::Mat& img, LetterboxInfo& info)
{
    float scale = std::min(static_cast<float>(kInputSize) / img.cols,
                            static_cast<float>(kInputSize) / img.rows);
    int unpadW = static_cast<int>(std::round(img.cols * scale));
    int unpadH = static_cast<int>(std::round(img.rows * scale));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(unpadW, unpadH), 0, 0, cv::INTER_LINEAR);

    int padW = kInputSize - unpadW;
    int padH = kInputSize - unpadH;
    int top = padH / 2;
    int bottom = padH - top;
    int left = padW / 2;
    int right = padW - left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT,
                        cv::Scalar(114, 114, 114));

    info.scale = scale;
    info.padLeft = left;
    info.padTop = top;
    info.unpadW = unpadW;
    info.unpadH = unpadH;
    return out;
}

std::vector<YoloSegDetector::Candidate> YoloSegDetector::NonMaxSuppression(
    std::vector<Candidate> dets)
{
    std::sort(dets.begin(), dets.end(),
              [](const Candidate& a, const Candidate& b) { return a.conf > b.conf; });
    std::vector<Candidate> kept;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;
            if (dets[i].classId != dets[j].classId) continue;  // per-class NMS
            if (IoU(dets[i], dets[j]) > kIouThresh) removed[j] = true;
        }
    }
    return kept;
}

YoloSegResult YoloSegDetector::Run(const cv::Mat& bgrFrame)
{
    YoloSegResult result;
    result.mask = cv::Mat::ones(bgrFrame.rows, bgrFrame.cols, CV_8UC1);

    // ---- Preprocess ----
    LetterboxInfo lb;
    cv::Mat letterboxed = Letterbox(bgrFrame, lb);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensorValues(3 * kInputSize * kInputSize);
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; c++) {
        channels[c] = cv::Mat(kInputSize, kInputSize, CV_32F,
                               inputTensorValues.data() + c * kInputSize * kInputSize);
    }
    cv::split(rgb, channels);

    std::array<int64_t, 4> inputShape = {1, 3, kInputSize, kInputSize};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(),
        inputShape.size());

    // ---- Inference ----
    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {output0Name_.c_str(), output1Name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                  outputNames, 2);

    // ---- Postprocess: decode output0 [1,116,8400] ----
    float* out0 = outputs[0].GetTensorMutableData<float>();
    auto out0Shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int numCandidates = static_cast<int>(out0Shape[2]);

    std::vector<Candidate> candidates;
    for (int i = 0; i < numCandidates; i++) {
        float cx = out0[0 * numCandidates + i];
        float cy = out0[1 * numCandidates + i];
        float w = out0[2 * numCandidates + i];
        float h = out0[3 * numCandidates + i];

        int bestClass = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < kNumClasses; c++) {
            float score = out0[(4 + c) * numCandidates + i];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestScore < kConfThresh) continue;
        // Skip candidates we'd throw away anyway (not a dynamic class) before
        // paying for mask-coefficient copies below.
        if (std::find(kDynamicClasses.begin(), kDynamicClasses.end(), bestClass) ==
            kDynamicClasses.end())
            continue;

        Candidate det;
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;
        det.x1 = (x1 - lb.padLeft) / lb.scale;
        det.y1 = (y1 - lb.padTop) / lb.scale;
        det.x2 = (x2 - lb.padLeft) / lb.scale;
        det.y2 = (y2 - lb.padTop) / lb.scale;
        det.classId = bestClass;
        det.conf = bestScore;
        det.maskCoeffs.resize(kNumMaskCoeffs);
        for (int m = 0; m < kNumMaskCoeffs; m++)
            det.maskCoeffs[m] = out0[(4 + kNumClasses + m) * numCandidates + i];
        candidates.push_back(std::move(det));
    }

    std::vector<Candidate> kept = NonMaxSuppression(std::move(candidates));

    // ---- Mask decode ----
    float* proto = outputs[1].GetTensorMutableData<float>();  // [1,32,160,160]

    for (const auto& det : kept) {
        cv::Mat mask160(kProtoSize, kProtoSize, CV_32F, cv::Scalar(0));
        for (int m = 0; m < kNumMaskCoeffs; m++) {
            float coeff = det.maskCoeffs[m];
            if (coeff == 0.0f) continue;
            cv::Mat protoPlane(kProtoSize, kProtoSize, CV_32F,
                                proto + m * kProtoSize * kProtoSize);
            mask160 += coeff * protoPlane;
        }
        for (int y = 0; y < kProtoSize; y++)
            for (int x = 0; x < kProtoSize; x++)
                mask160.at<float>(y, x) = Sigmoid(mask160.at<float>(y, x));

        cv::Mat mask640;
        cv::resize(mask160, mask640, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);

        cv::Rect unpadRoi(lb.padLeft, lb.padTop, lb.unpadW, lb.unpadH);
        unpadRoi &= cv::Rect(0, 0, mask640.cols, mask640.rows);
        cv::Mat maskUnpad = mask640(unpadRoi);

        cv::Mat maskFull;
        cv::resize(maskUnpad, maskFull, cv::Size(bgrFrame.cols, bgrFrame.rows), 0, 0,
                   cv::INTER_LINEAR);

        cv::Rect boxRoi(static_cast<int>(std::round(det.x1)),
                         static_cast<int>(std::round(det.y1)),
                         static_cast<int>(std::round(det.x2 - det.x1)),
                         static_cast<int>(std::round(det.y2 - det.y1)));
        boxRoi &= cv::Rect(0, 0, bgrFrame.cols, bgrFrame.rows);
        if (boxRoi.width <= 0 || boxRoi.height <= 0) continue;

        cv::Mat maskFullRoi = maskFull(boxRoi);
        cv::Mat resultRoi = result.mask(boxRoi);
        resultRoi.setTo(0, maskFullRoi > 0.5f);

        ORB_SLAM3::Detection outDet;
        outDet.bbox = boxRoi;
        outDet.class_id = det.classId;
        outDet.confidence = det.conf;
        result.detections.push_back(outDet);
    }

    return result;
}

}  // namespace ORB_SLAM3_Wrapper
