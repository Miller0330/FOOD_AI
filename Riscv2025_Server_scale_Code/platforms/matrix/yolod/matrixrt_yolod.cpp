// matrixrt_yolod - resident variant of the SDK matrixrt_yolo example.
// Loads the model once, then serves requests from stdin:
//   <image_path>[TAB<annotated_out_path>]\n
// Replies with the same "#N  class=.. (..)  score=..  bbox=[..]" lines as
// matrixrt_yolo, then "@@END <infer_ms> <total_ms>\n" (or "@@ERR <msg>\n").
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "matrix/matrixrt.hpp"

using namespace matrixrt;
namespace {
int MODEL_W = 512, MODEL_H = 512, NUM_CLASSES = 100, NUM_ANCHORS = 0;
bool LETTERBOX = false, MULTI = false, REMAP = false;
struct BBox { float x1, y1, x2, y2, score; int class_id; };

std::shared_ptr<uint8_t> page_alloc(size_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>((uint8_t*)p, [size](void* x) { munmap(x, size); });
}
void preprocess(const cv::Mat& bgr, uint8_t* out) {
    cv::Mat r, rgb;
    if (LETTERBOX) {
        float sc = std::min((float)MODEL_W / bgr.cols, (float)MODEL_H / bgr.rows);
        int nw = (int)std::lround(bgr.cols * sc), nh = (int)std::lround(bgr.rows * sc);
        r = cv::Mat(MODEL_H, MODEL_W, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat small;
        cv::resize(bgr, small, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        small.copyTo(r(cv::Rect((MODEL_W - nw) / 2, (MODEL_H - nh) / 2, nw, nh)));
    } else {
        cv::resize(bgr, r, cv::Size(MODEL_W, MODEL_H), 0, 0, cv::INTER_LINEAR);
    }
    cv::cvtColor(r, rgb, cv::COLOR_BGR2RGB);
    size_t plane = (size_t)MODEL_W * MODEL_H;
    std::vector<cv::Mat> ch; cv::split(rgb, ch);
    for (int i = 0; i < 3; ++i) memcpy(out + i * plane, ch[i].data, plane);
}
float iou(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6f);
}
std::vector<BBox> nms(std::vector<BBox>& v, float t) {
    std::sort(v.begin(), v.end(), [](const BBox& a, const BBox& b) { return a.score > b.score; });
    std::vector<bool> sup(v.size(), false); std::vector<BBox> keep;
    for (size_t i = 0; i < v.size(); ++i) {
        if (sup[i]) continue; keep.push_back(v[i]);
        for (size_t j = i + 1; j < v.size(); ++j)
            if (!sup[j] && v[i].class_id == v[j].class_id && iou(v[i], v[j]) > t) sup[j] = true;
    }
    return keep;
}
std::vector<BBox> decode(const float* d, float conf, float iou_t) {
    std::vector<BBox> cand;
    for (int a = 0; a < NUM_ANCHORS; ++a) {
        float cx = d[a], cy = d[NUM_ANCHORS + a], w = d[2 * NUM_ANCHORS + a], h = d[3 * NUM_ANCHORS + a];
        auto push = [&](float sc, int c) {
            cand.push_back({std::max(0.f, cx - w / 2), std::max(0.f, cy - h / 2),
                            std::min((float)MODEL_W, cx + w / 2), std::min((float)MODEL_H, cy + h / 2), sc, c});
        };
        if (MULTI) {
            for (int c = 0; c < NUM_CLASSES; ++c) { float s = d[(4 + c) * NUM_ANCHORS + a]; if (s >= conf) push(s, c); }
        } else {
            float ms = 0.f; int mc = 0;
            for (int c = 0; c < NUM_CLASSES; ++c) { float s = d[(4 + c) * NUM_ANCHORS + a]; if (s > ms) { ms = s; mc = c; } }
            if (ms >= conf) push(ms, mc);
        }
    }
    auto kept = nms(cand, iou_t);
    if (MULTI && kept.size() > 300) kept.resize(300);
    return kept;
}
std::vector<std::string> load_labels(const std::string& p) {
    std::vector<std::string> o; std::ifstream f(p); std::string l;
    while (std::getline(f, l)) { while (!l.empty() && l.back() == '\r') l.pop_back(); o.push_back(l); }
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "matrixrt_yolod <model.nb> [-c labels] [--imgsz n] [--nc n] [--cores n] [--conf f] [--iou f]\n"; return 1; }
    std::string model = argv[1], labels_path; float conf = 0.15f, iou_t = 0.45f; int cores = 2;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-c" && i + 1 < argc) labels_path = argv[++i];
        else if (a == "--imgsz" && i + 1 < argc) MODEL_W = MODEL_H = std::stoi(argv[++i]);
        else if (a == "--nc" && i + 1 < argc) NUM_CLASSES = std::stoi(argv[++i]);
        else if (a == "--cores" && i + 1 < argc) cores = std::stoi(argv[++i]);
        else if (a == "--conf" && i + 1 < argc) conf = std::stof(argv[++i]);
        else if (a == "--iou" && i + 1 < argc) iou_t = std::stof(argv[++i]);
        else if (a == "--letterbox") LETTERBOX = true;
        else if (a == "--multi") MULTI = true;
        else if (a == "--remap") REMAP = true;
        else { std::cerr << "unknown flag " << a << "\n"; return 2; }
    }
    try {
        auto all = Device::scan().expect("scan");
        if (cores > (int)all.size()) cores = (int)all.size();
        std::vector<std::string> use(all.begin(), all.begin() + cores);
        auto vd = VDevice::create(use).expect("VDevice::create");
        auto im = vd->create_infer_model(model.c_str()).expect("create_infer_model");
        std::string in_name = im->get_input_names()[0], out_name = im->get_output_names()[0];
        im->set_power_mode(MATRIX_POWER_MODE_ULTRA_PERFORMANCE);
        im->output(out_name)->set_format_type(MATRIX_FORMAT_TYPE_FLOAT32);
        size_t in_size = im->input(in_name)->get_frame_size(), out_size = im->output(out_name)->get_frame_size();
        int ch = 4 + NUM_CLASSES;
        if (out_size % (sizeof(float) * ch) != 0 || in_size != (size_t)3 * MODEL_W * MODEL_H) {
            std::cerr << "model geometry mismatch in=" << in_size << " out=" << out_size << "\n"; return 1;
        }
        NUM_ANCHORS = (int)(out_size / (sizeof(float) * ch));
        auto configured = im->configure().expect("configure");
        auto bindings = configured.create_bindings().expect("bindings");
        auto in_buf = page_alloc(in_size), out_buf = page_alloc(out_size);
        if (bindings.input(in_name)->set_buffer(MemoryView(in_buf.get(), in_size)) != MATRIX_SUCCESS ||
            bindings.output(out_name)->set_buffer(MemoryView(out_buf.get(), out_size)) != MATRIX_SUCCESS) {
            std::cerr << "set buffer failed\n"; return 1;
        }
        std::vector<std::string> labels; if (!labels_path.empty()) labels = load_labels(labels_path);
        std::cout << "@@READY cores=" << cores << " anchors=" << NUM_ANCHORS << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line == "quit") break;
            auto T0 = std::chrono::steady_clock::now();
            std::string img = line, outp; auto tab = line.find('\t');
            if (tab != std::string::npos) { img = line.substr(0, tab); outp = line.substr(tab + 1); }
            cv::Mat bgr = cv::imread(img, cv::IMREAD_COLOR);
            double t_read = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            if (bgr.empty()) { std::cout << "@@ERR cannot read image " << img << std::endl; continue; }
            try {
                auto tp0 = std::chrono::steady_clock::now();
                preprocess(bgr, in_buf.get());
                auto t0 = std::chrono::steady_clock::now();
                double t_pre = std::chrono::duration<double, std::milli>(t0 - tp0).count();
                auto job = configured.run_async(bindings).expect("run_async");
                if (job.wait(std::chrono::milliseconds(10000)) != MATRIX_SUCCESS) throw std::runtime_error("wait timeout");
                double inf = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                auto td0 = std::chrono::steady_clock::now();
                auto dets = decode((const float*)out_buf.get(), conf, iou_t);
                if (LETTERBOX && REMAP) {
                    // letterbox space -> original image stretched to MODEL_WxMODEL_H (what the web overlay expects)
                    float sc = std::min((float)MODEL_W / bgr.cols, (float)MODEL_H / bgr.rows);
                    int nw = (int)std::lround(bgr.cols * sc), nh = (int)std::lround(bgr.rows * sc);
                    float ox = (float)((MODEL_W - nw) / 2), oy = (float)((MODEL_H - nh) / 2);
                    for (auto& b : dets) {
                        auto fx = [&](float v) { return std::min((float)MODEL_W, std::max(0.f, (v - ox) / sc * MODEL_W / bgr.cols)); };
                        auto fy = [&](float v) { return std::min((float)MODEL_H, std::max(0.f, (v - oy) / sc * MODEL_H / bgr.rows)); };
                        b.x1 = fx(b.x1); b.x2 = fx(b.x2); b.y1 = fy(b.y1); b.y2 = fy(b.y2);
                    }
                }
                double t_dec = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                for (size_t i = 0; i < dets.size(); ++i) {
                    auto& b = dets[i];
                    std::string nm = b.class_id < (int)labels.size() ? labels[b.class_id] : "class_" + std::to_string(b.class_id);
                    printf("  #%zu  class=%-3d (%s)  score=%.3f  bbox=[%.1f,%.1f -> %.1f,%.1f]\n", i + 1, b.class_id, nm.c_str(), b.score, b.x1, b.y1, b.x2, b.y2);
                }
                if (!outp.empty()) {
                    cv::Mat canvas; cv::resize(bgr, canvas, cv::Size(MODEL_W, MODEL_H));
                    for (auto& b : dets) cv::rectangle(canvas, cv::Rect((int)b.x1, (int)b.y1, (int)(b.x2 - b.x1), (int)(b.y2 - b.y1)), cv::Scalar(0, 255, 0), 2);
                    cv::imwrite(outp, canvas);
                }
                double tot = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
                printf("@@END %.2f %.2f read=%.2f pre=%.2f dec=%.2f\n", inf, tot, t_read, t_pre, t_dec); fflush(stdout);
            } catch (const std::exception& e) { std::cout << "@@ERR " << e.what() << std::endl; }
        }
    } catch (const std::exception& e) { std::cerr << "fatal: " << e.what() << "\n"; return 1; }
    return 0;
}
