/**
 * 同步捕捉接口
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <tuple>


namespace cap_core {

    using std::vector;
    using std::tuple;
    using cv::Mat;

    /**
     * 多视图捕捉的结果
     */
    struct CaptureResult {
        int cap_ix; // 序号
        std::size_t n_device{0}; // 总视角数量
        std::size_t n_success{0}; // 成功的捕捉数量

        // 捕捉成功 flag、图像 width、图像 height、图像数据
        vector<tuple<bool, int, int, Mat>> data;
    };


    /**
     * TODO
     * 同步捕捉接口
     */
    class Int_SyncCapture {
    public:
        /**
         * 初始化接口，调用此方法进行参数的初始化
         */
        virtual void init();
        /**
         * 捕捉接口，调用此方法使捕捉系统捕获同步图像，并将捕捉结果存储在 CaptureResult 中
         * @return 捕捉结果
         */
        virtual CaptureResult capture() = 0;
        /**
         * 释放资源
         */
        virtual void release();
    };

}
