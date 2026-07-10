// Standalone prototype: YOLOv8-seg inference via ONNX Runtime C++ API,
// replicating yolo_node.py's preprocessing/postprocessing so the output can
// be diffed against the Python (ultralytics) reference on the same frame.
// Not wired into ORB_SLAM3 yet - this only exists to de-risk the pre/post
// processing math before committing to the full rgbd-slam-node.cpp rewrite.
//
// Model I/O (confirmed via onnxruntime python introspection):
//   input  "images"  [1,3,640,640] float32
//   output "output0" [1,116,8400]  float32   (4 bbox + 80 class scores + 32 mask coeffs)
//   output "output1" [1,32,160,160] float32  (mask prototypes)

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kNumClasses = 80;
constexpr int kNumMaskCoeffs = 32;
constexpr int kProtoSize = 160;
constexpr float kConfThresh = 0.25f;
constexpr float kIouThresh = 0.7f;

const std::vector<int> kDynamicClasses = {0, 1, 2, 3, 5, 7};

struct LetterboxInfo {
    float scale;
    int padLeft, padTop;
    int unpadW, unpadH;
};

// Same convention as ultralytics' LetterBox: resize keeping aspect ratio to
// fit inside kInputSize x kInputSize, pad the rest with (114,114,114), pad
// split evenly on both sides.
cv::Mat Letterbox(const cv::Mat& img, LetterboxInfo& info) {
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
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    info.scale = scale;
    info.padLeft = left;
    info.padTop = top;
    info.unpadW = unpadW;
    info.unpadH = unpadH;
    return out;
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Detection {
    float x1, y1, x2, y2;  // original-image pixel coords
    int classId;
    float conf;
    std::array<float, kNumMaskCoeffs> maskCoeffs;
};

float IoU(const Detection& a, const Detection& b) {
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

std::vector<Detection> NMS(std::vector<Detection> dets) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<Detection> kept;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;
            // Per-class NMS (matches ultralytics default agnostic=False).
            if (dets[i].classId != dets[j].classId) continue;
            if (IoU(dets[i], dets[j]) > kIouThresh) removed[j] = true;
        }
    }
    return kept;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.onnx> <image.png> [out_dir] [--gpu]\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string outDir = ".";
    bool useGpu = false;
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gpu") useGpu = true;
        else outDir = arg;
    }

    cv::Mat frame = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "failed to read image: " << imagePath << "\n";
        return 1;
    }
    std::cout << "frame size: " << frame.cols << "x" << frame.rows << "\n";

    // ---- Preprocess ----
    LetterboxInfo lb;
    cv::Mat letterboxed = Letterbox(frame, lb);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> inputTensorValues(3 * kInputSize * kInputSize);
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; c++) {
        channels[c] = cv::Mat(kInputSize, kInputSize, CV_32F,
                               inputTensorValues.data() + c * kInputSize * kInputSize);
    }
    cv::split(rgb, channels);

    // ---- ONNX Runtime session ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_seg_prototype");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    if (useGpu) {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "requested CUDA execution provider\n";
    }
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAlloc = session.GetInputNameAllocated(0, allocator);
    auto output0NameAlloc = session.GetOutputNameAllocated(0, allocator);
    auto output1NameAlloc = session.GetOutputNameAllocated(1, allocator);
    std::vector<const char*> inputNames = {inputNameAlloc.get()};
    std::vector<const char*> outputNames = {output0NameAlloc.get(), output1NameAlloc.get()};

    std::array<int64_t, 4> inputShape = {1, 3, kInputSize, kInputSize};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    // Warm-up (CUDA EP does cuDNN algo autotune + context init on first call).
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
                                outputNames.data(), outputNames.size());

    constexpr int kTimedRuns = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedRuns; i++) {
        outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
                               outputNames.data(), outputNames.size());
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / kTimedRuns;
    std::cout << "inference took " << ms << " ms (avg of " << kTimedRuns << " runs)\n";

    // ---- Postprocess: decode output0 [1,116,8400] ----
    float* out0 = outputs[0].GetTensorMutableData<float>();
    auto out0Shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1,116,8400]
    int numAttrs = static_cast<int>(out0Shape[1]);   // 116
    int numCandidates = static_cast<int>(out0Shape[2]);  // 8400

    std::vector<Detection> candidates;
    for (int i = 0; i < numCandidates; i++) {
        // out0 is [attr, candidate] row-major over attr - element (a, i) is at out0[a*numCandidates + i]
        float cx = out0[0 * numCandidates + i];
        float cy = out0[1 * numCandidates + i];
        float w  = out0[2 * numCandidates + i];
        float h  = out0[3 * numCandidates + i];

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

        Detection det;
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;
        // Un-letterbox: 640-space -> original image space.
        det.x1 = (x1 - lb.padLeft) / lb.scale;
        det.y1 = (y1 - lb.padTop) / lb.scale;
        det.x2 = (x2 - lb.padLeft) / lb.scale;
        det.y2 = (y2 - lb.padTop) / lb.scale;
        det.classId = bestClass;
        det.conf = bestScore;
        for (int m = 0; m < kNumMaskCoeffs; m++)
            det.maskCoeffs[m] = out0[(4 + kNumClasses + m) * numCandidates + i];
        candidates.push_back(det);
    }
    std::cout << "candidates above conf thresh: " << candidates.size() << "\n";

    std::vector<Detection> kept = NMS(std::move(candidates));
    std::cout << "after NMS: " << kept.size() << "\n";

    // ---- Mask decode ----
    float* proto = outputs[1].GetTensorMutableData<float>();  // [1,32,160,160]

    cv::Mat dynamicMask = cv::Mat::ones(frame.rows, frame.cols, CV_8UC1);
    std::ofstream detLog(outDir + "/cpp_detections.json");
    detLog << "[\n";
    bool first = true;

    for (const auto& det : kept) {
        if (std::find(kDynamicClasses.begin(), kDynamicClasses.end(), det.classId) ==
            kDynamicClasses.end())
            continue;

        // mask160 = sigmoid(coeffs . proto_flat), proto_flat is [32, 160*160]
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

        // Upsample to the 640x640 letterboxed space, then crop out the
        // letterbox padding, then resize to the original frame size - same
        // coordinate chain used to un-letterbox the boxes above.
        cv::Mat mask640;
        cv::resize(mask160, mask640, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);

        cv::Rect unpadRoi(lb.padLeft, lb.padTop, lb.unpadW, lb.unpadH);
        unpadRoi &= cv::Rect(0, 0, mask640.cols, mask640.rows);
        cv::Mat maskUnpad = mask640(unpadRoi);

        cv::Mat maskFull;
        cv::resize(maskUnpad, maskFull, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);

        // Zero out anything outside this detection's own box (ultralytics
        // does the equivalent crop-to-box step).
        cv::Rect boxRoi(
            static_cast<int>(std::round(det.x1)), static_cast<int>(std::round(det.y1)),
            static_cast<int>(std::round(det.x2 - det.x1)), static_cast<int>(std::round(det.y2 - det.y1)));
        boxRoi &= cv::Rect(0, 0, frame.cols, frame.rows);

        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                bool inBox = boxRoi.contains(cv::Point(x, y));
                if (inBox && maskFull.at<float>(y, x) > 0.5f)
                    dynamicMask.at<uchar>(y, x) = 0;
            }
        }

        if (!first) detLog << ",\n";
        first = false;
        detLog << "  {\"class_id\": " << det.classId << ", \"conf\": " << det.conf
               << ", \"bbox\": [" << (int)det.x1 << ", " << (int)det.y1 << ", " << (int)det.x2
               << ", " << (int)det.y2 << "]}";

        std::cout << "det: class=" << det.classId << " conf=" << det.conf << " bbox=(" << det.x1
                  << "," << det.y1 << "," << det.x2 << "," << det.y2 << ")\n";
    }
    detLog << "\n]\n";
    detLog.close();

    cv::imwrite(outDir + "/cpp_mask.png", dynamicMask * 255);

    int dynamicPixels = cv::countNonZero(dynamicMask == 0);
    std::cout << "dynamic pixel count: " << dynamicPixels << " / " << dynamicMask.total() << "\n";

    return 0;
}
