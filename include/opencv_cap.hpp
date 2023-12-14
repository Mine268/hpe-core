#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "sync_cap_core.hpp"


namespace cap_core {

    class SyncOpenCVCap : public virtual Int_SyncCapture {
    private:
        vector<cv::VideoCapture> sources;
        int cap_ix;

    public:
        /**
         * 初始化 opencv 捕捉系统
         * @param device_ixs 捕捉的设备编号列表
         */
        explicit SyncOpenCVCap(const vector<int>& device_ixs);
        CaptureResult capture() override;
        void release() override;
    };

}
