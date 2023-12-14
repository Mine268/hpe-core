#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <tuple>
#include <string>
#include <map>


namespace hpe_core {

    using std::vector;
    using std::tuple;
    using std::string;
    using std::map;

    using cv::Point3f;
    using cv::Point2f;
    using cv::Mat;

    /**
     * @brief 人体姿态参数结构体，建议使用工厂模式针对不同的人体结构初始化对应的 body_connection 和 hand_connections 以及对应的描述
     * body_joint_desc 和 hand_joint_desc
     */
    struct HumanPose {
        // 全身的关节点位置
        vector<Point3f> kps;

        // 人体和人手的关节点连接方式
        vector<tuple<int, int>> body_connection;
        vector<tuple<int, int>> hand_connections;

        // 人体和人手的关节点说明
        map<int, string> body_joint_desc;
        map<int, string> hand_joint_desc;
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

        /**
         * @brief 初始化人体姿态数据体，按照所使用的姿态估计器的模型所使用的人体结构初始化人体结构体。包括 \n
         * - body_connection, hand_connections 人体关节的连接方式 \n
         * - body_joint_desc, hand_joint_desc 人体关节的名称
         * @return 初始化之后的人体关节结构体
         */
        virtual HumanPose init_struct() = 0;
    };


    /**
     * @brief 单人三维人体姿态估计接口
     * @note 单人三维人体姿态估计需要组合单人二维姿态估计的接口，传入多视图图像，输出单人三维姿态
     */
    class Int_Single3dHPE {
    public:
        explicit Int_Single3dHPE(Int_Single2dHPE* hpe_2d);

        /**
         * @brief 估计单个的三维人体姿态，相机坐标系变换 rot * x + tran
         * @param multiview_img 单帧多视图图像
         * @param intrs 相机内参
         * @param rots 相机旋转矩阵
         * @param trans 相机平移向量
         * @return 当前场景中的三维人体姿态
         */
        virtual HumanPose predict(
                const vector<Mat>& multiview_img,
                const vector<Mat>& intrs,
                const vector<Mat>& rots,
                const vector<Mat>& trans
                ) = 0;
        /**
         * @brief 估计单个的三维人体姿态，相机坐标系变换 rot * x + tran
         * @param multiview_imgs 一组多视图图像
         * @param batch_intrs 相机内参
         * @param batch_rots 相机旋转矩阵
         * @param batch_trans 相机平移向量
         * @return 一组三维人体姿态
         */
        virtual vector<HumanPose> predict(
                const vector<vector<Mat>>& multiview_imgs,
                const vector<vector<Mat>>& batch_intrs,
                const vector<vector<Mat>>& batch_rots,
                const vector<vector<Mat>>& batch_trans
                ) = 0;

    protected:
        Int_Single2dHPE* hpe_2d;
    };
}
