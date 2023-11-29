#include "hrnet_2d.hpp"

#include <cassert>
#include <utility>


hpe_core::HRNet_2d::HRNet_2d(std::string onnx_path) {
    this->onnx_path = std::move(onnx_path);
    this->net = cv::dnn::readNetFromONNX(this->onnx_path);
}

std::tuple<std::vector<cv::Point2f>, std::vector<float>>
hpe_core::HRNet_2d::predict(const cv::Mat &img) {
    cv::Mat blob = cv::dnn::blobFromImage(
            img,
            1 / 58.,
            cv::Size{this->input_hight, this->input_width},
            cv::Scalar{123.675, 116.28, 103.53},
            false, false
            );

    this->net.setInput(blob);
    auto pred_hm = this->net.forward();

    vector<Point2f> joints;
    vector<float> conf;
    int pos_ixs[4] {0};
    for (int jx = 0; jx < pred_hm.size[1]; ++jx) {
        float posX{ 0 }, posY{ 0 }, sum{ 0 };
        pos_ixs[1] = jx;
        for (int h = 0; h < pred_hm.size[2]; ++h) {
            pos_ixs[2] = h;
            for (int w = 0; w < pred_hm.size[3]; ++w) {
                pos_ixs[3] = w;
                sum += pred_hm.at<float>(pos_ixs);
                posX += pred_hm.at<float>(pos_ixs) * static_cast<float>(h);
                posY += pred_hm.at<float>(pos_ixs) * static_cast<float>(w);
            }
        }
        posX /= sum;
        posY /= sum;
        joints.emplace_back(posY, posX);
    }

    for (int jx = 0; jx < joints.size(); ++jx) {
        pos_ixs[1] = jx;
        pos_ixs[2] = static_cast<int>(joints[jx].y);
        pos_ixs[3] = static_cast<int>(joints[jx].x);
        conf.push_back(pred_hm.at<float>(pos_ixs));
        joints[jx].y *= static_cast<float>(img.size[0]) / static_cast<float>(pred_hm.size[2]);
        joints[jx].x *= static_cast<float>(img.size[1]) / static_cast<float>(pred_hm.size[3]);
    }

    return {joints, conf};
}

std::tuple<std::vector<std::vector<cv::Point2f>>, std::vector<std::vector<float>>>
hpe_core::HRNet_2d::predict(const std::vector<cv::Mat> &imgs) {
    cv::Mat blobs = cv::dnn::blobFromImages(
            imgs,
            1 / 58.,
            cv::Size(this->input_hight, this->input_width),
            cv::Scalar{123.675, 116.28, 103.53},
            false, false
            );

    this->net.setInput(blobs);
    auto pred_hms = this->net.forward();

    vector<vector<Point2f>> joint_ss;
    vector<vector<float>> conf_ss;
    int pos_ixs[4] {0};
    for (int bx = 0; bx < pred_hms.size[0]; ++bx) {
        vector<Point2f> joints;
        for (int jx = 0; jx < pred_hms.size[1]; ++jx) {
            float posX{0}, posY{0}, sum{0};
            for (int h = 0; h < pred_hms.size[2]; ++h) {
                for (int w = 0; w < pred_hms.size[3]; ++w) {
                    pos_ixs[0] = bx;
                    pos_ixs[1] = jx;
                    pos_ixs[2] = h;
                    pos_ixs[3] = w;
                    sum += pred_hms.at<float>(pos_ixs);
                    posX += pred_hms.at<float>(pos_ixs) * static_cast<float>(h);
                    posY += pred_hms.at<float>(pos_ixs) * static_cast<float>(w);
                }
            }
            posX /= sum;
            posY /= sum;
            joints.emplace_back(posY, posX);
        }
        joint_ss.emplace_back(std::move(joints));
    }

    for (int bx = 0; bx < pred_hms.size[0]; ++bx) {
        vector<float> conf;
        for (int jx = 0; jx < pred_hms.size[1]; ++jx) {
            pos_ixs[0] = bx;
            pos_ixs[1] = jx;
            pos_ixs[2] = static_cast<int>(joint_ss[bx][jx].y);
            pos_ixs[3] = static_cast<int>(joint_ss[bx][jx].x);
            conf.push_back(pred_hms.at<float>(pos_ixs));
            joint_ss[bx][jx].y *= static_cast<float>(imgs[0].size[0]) / static_cast<float>(pred_hms.size[2]);
            joint_ss[bx][jx].x *= static_cast<float>(imgs[0].size[1]) / static_cast<float>(pred_hms.size[3]);
        }
    }

    return {joint_ss, conf_ss};
}
