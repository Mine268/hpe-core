#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <tuple>
#include <string>


namespace hpe_core {

    using std::vector;
    using std::tuple;
    using std::string;

    using cv::Point3f;
    using cv::Point2f;
    using cv::Mat;

    /**
     * @brief 人体姿态参数结构体，建议使用工厂模式针对不同的人体结构初始化对应的 body_connection 和 hand_connections 以及对应的描述
     * body_joint_desc 和 hand_joint_desc
     */
    struct HumanPose {
        // 人体和人手的关节点坐标
        vector<Point3f> body_joints;
        vector<Point3f> hand_joints;

        // 人体和人手的关节点连接方式
        vector<tuple<int, int>> body_connection;
        vector<tuple<int, int>> hand_connections;

        // 人体和人手的关节点说明
        vector<string> body_joint_desc;
        vector<string> hand_joint_desc;
    };

    /**
     * @brief 利用多视图的二维人体姿态，结合多视图的相机位姿、内部参数，计算三维人体姿态，其中代表相机位姿的参数 rots 和 trans 对
     *  世界坐标 x 转换为相机坐标的变换为 rots * x + trans
     * @param points 多视图的二维人体姿态预测结果，第一层 vector 中的每个元素是每个视图的二维人体姿态估计结果，第二层 vector 中的每
     *  个元素代表了这个预测结果中每个关节点在图像上的二维坐标位置
     * @param intrs 每个视图的内部参数
     * @param rots 每个视图的外部参数中代表旋转的矩阵
     * @param trans 每个视图的外部参数中代表位移的矩阵
     * @return
     */
    vector<Point3f> triangulatePoints(
            const vector<vector<Point2f>>& points, // [V,J,2]
            const vector<Mat>& intrs,
            const vector<Mat>& rots,
            const vector<Mat>& trans // Rx+t
    );


    /**
     * @brief 单人二维人体姿态估计接口
     */
    class Int_Single2dHPE {
    public:
        /**
         * @brief 估计单张图片中的二维人体姿态
         * @param img 有待估计姿态的图片
         * @return 二维人体姿态估计的结果，以及结果的置信度
         */
        virtual tuple<vector<Point2f>, vector<float>> predict(const Mat& img) = 0;
        /**
         * @brief 估计多个视图中的二维人体姿态
         * @param imgs 有待估计的姿态的多视图图片
         * @return 多张图片中的姿态估计结果，以及结果的置信度
         */
        virtual tuple<vector<vector<Point2f>>, vector<vector<float>>> predict(const vector<Mat>& imgs) = 0;
    };
}
