#pragma once

#include <opencv2/opencv.hpp>
#include "hpe_core.hpp"


namespace hpe_core {

    /**
     * @brief 采用 COCO Wholebody 训练的 hrnet 进行关键点位置 2D 估计
     */
    class HRNet_2d : Int_Single2dHPE {
    private:
        /**
         * 模型 onnx 文件地址
         */
        std::string onnx_path;

        int input_hight = 384;
        int input_width = 288;
        int output_hight = 96;
        int output_width = 72;

        cv::dnn::Net net;

    public:
        /**
         * @param onnx_path 模型 onnx 文件地址
         */
        explicit HRNet_2d(std::string onnx_path);

        tuple<vector<Point2f>, vector<float>> predict(const Mat& img) override;

        /**
         * @param imgs 数组中的图片大小必须相同
         * @return
         */
        tuple<vector<vector<Point2f>>, vector<vector<float>>> predict(const vector<Mat>& imgs) override;

        HumanPose init_struct() override;
    };

}
