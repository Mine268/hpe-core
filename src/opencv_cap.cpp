#include "opencv_cap.hpp"


cap_core::SyncOpenCVCap::SyncOpenCVCap(const vector<int>& device_ixs) {
    cap_ix = 0;
    for (auto ix : device_ixs) {
        sources.emplace_back(ix);
    }
}

void cap_core::SyncOpenCVCap::release() {
    for (auto& source : sources) {
        source.release();
    }
}

cap_core::CaptureResult cap_core::SyncOpenCVCap::capture() {
    CaptureResult ret;

    ret.n_device = sources.size();
    // TODO 循环捕捉十个以内的摄像头的画面，延迟应该能够满足同步捕捉的要求，但是更多的摄像头可能需要使用其他技术了
    for (auto& source: sources) {
        Mat img;
        bool flag = false;
        if (source.isOpened()) {
            flag = source.read(img);
        }
        ret.cap_ix = (this->cap_ix)++;
        if (flag) {
            ++(ret.n_success);
        }
        ret.data.emplace_back(
                flag,
                img.size[1],
                img.size[0],
                img
                );
    }

    return ret;
}


