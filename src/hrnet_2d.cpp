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

hpe_core::HumanPose hpe_core::HRNet_2d::init_struct() {
    HumanPose pose_struct;

    pose_struct.body_connection = {
            {1, 2}, {1, 3}, {2, 4}, {3, 5},
            {1, 6}, {6, 8}, {8, 10},
            {1, 7}, {7, 9}, {9, 11},
            {6, 12}, {12, 14}, {14, 16},
            {7, 13}, {13, 15}, {15, 17},
            {17, 23}, {17, 21}, {17, 22},
            {16, 20}, {16, 19}, {16, 18}
    };

    pose_struct.hand_connections = {
            // right
            {113, 114}, {114, 115}, {115, 116}, {116, 117},
            {113, 118}, {118, 119}, {119, 120}, {120, 121},
            {113, 122}, {122, 123}, {123, 124}, {124, 125},
            {113, 126}, {126, 127}, {127, 128}, {128, 129},
            {113, 130}, {130, 131}, {131, 132}, {132, 133},
            // left
            {92, 93}, {93, 94}, {94, 95}, {95, 96},
            {92, 97}, {97, 98}, {98, 99}, {99, 100},
            {92, 101}, {101, 102}, {102, 103}, {103, 104},
            {92, 105}, {105, 106}, {106, 107}, {107, 108},
            {92, 109}, {109, 110}, {110, 111}, {111, 112}
    };

    pose_struct.body_joint_desc = {
            {1, "head"},
            {2, "left_eye"}, {3, "right_eye"},
            {4, "left_ear"}, {5, "right_ear"},
            {6, "left_shoulder"}, {7, "right_shoulder"},
            {8, "left_elbow"}, {9, "right_elbow"},
            {10, "left_wrist"}, {11, "right_wrist"},
            {12, "left_pelvis"}, {13, "right_pelvis"},
            {14, "left_ankle"}, {15, "right_ankle"},
            {16, "left_foot"}, {17, "right_foot"},
            {20, "left_foot_bottom"}, {23, "right_foot_bottom"},
            {18, "left_foot_toe"}, {21, "right_foot_toe"},
            {19, "left_foot_little_toe"}, {22, "right_foot_little_toe"}
    };

    pose_struct.hand_joint_desc = {
            // left
            {92, "left_hand_wrist"},
            {93, "left_thumb_0"}, {94, "left_thumb_1"}, {95, "left_thumb_2"}, {96, "left_thumb_3"},
            {97, "left_index_0"}, {98, "left_index_1"}, {99, "left_index_2"}, {100, "left_index_3"},
            {101, "left_middle_0"}, {102, "left_middle_1"}, {103, "left_middle_2"}, {104, "left_middle_3"},
            {105, "left_ring_0"}, {106, "left_ring_1"}, {107, "left_ring_2"}, {108, "left_ring_3"},
            {109, "left_pinky_0"}, {110, "left_pinky_1"}, {111, "left_pinky_2"}, {112, "left_pinky_3"},
            // right
            {113, "right_hand_wrist"},
            {114, "right_thumb_0"}, {115, "right_thumb_1"}, {116, "right_thumb_2"}, {117, "right_thumb_3"},
            {118, "right_index_0"}, {119, "right_index_1"}, {120, "right_index_2"}, {121, "right_index_3"},
            {122, "right_middle_0"}, {123, "right_middle_1"}, {124, "right_middle_2"}, {125, "right_middle_3"},
            {126, "right_ring_0"}, {127, "right_ring_1"}, {128, "right_ring_2"}, {129, "right_ring_3"},
            {130, "right_pinky_0"}, {131, "right_pinky_1"},  {132, "right_pinky_2"}, {133, "right_pinky_3"}
    };

    return pose_struct;
}
