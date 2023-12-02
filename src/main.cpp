#include <iostream>
#include "hrnet_2d.hpp"


int main() {

    hpe_core::HRNet_2d net {R"(../model/hrnet.onnx)"};

    auto img1 = cv::imread("../example/clean1.png");
    cv::Mat img2;

    cv::flip(img1, img2, 1);
    std::vector<cv::Mat> input = {img1, img2};

    auto [j_ss, conf_ss] = net.predict(input);

    for (auto jx : j_ss[0]) {
        cv::circle(img1, jx, 3, cv::Scalar{0, 0, 255}, 3);
    }
    for (auto jx : j_ss[1]) {
        cv::circle(img2, jx, 3, cv::Scalar{0, 0, 255}, 3);
    }

    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    cv::waitKey(0);

//    auto [js, cs] = net.predict(img1);
//    for (auto jn : js) {
//        cv::circle(img1, jn, 3, cv::Scalar{0, 0, 255}, 3);
//    }
//    cv::imshow("img1", img1);
//    cv::waitKey(0);

    return 0;
}
