#include "hpe_core.hpp"


namespace hpe_core {
    [[maybe_unused]] vector<Point3f> triangulatePoints(
            const vector<vector<Point2f>>& points, // [V,J,2]
            const vector<Mat>& intrs,
            const vector<Mat>& rots,
            const vector<Mat>& trans) {
        auto n_view = points.size();
        CV_Assert(n_view >= 2 &&
                  n_view == intrs.size() && n_view == rots.size() && n_view == trans.size());
        auto n_joint = points[0].size();
        CV_Assert(n_joint >= 1);

        std::vector<cv::Mat> proj_mats;
        for (int i = 0; i < n_view; ++i) {
            cv::Mat proj_mat;
            cv::hconcat(rots[i], trans[i], proj_mat);
            proj_mat = intrs[i] * proj_mat;
            proj_mats.push_back(proj_mat);
        }

        std::vector<cv::Point3f> points3d;
        cv::Mat_<double> constraint_mat(
                2 * static_cast<int>(n_view),
                4);

        for (int jx = 0; jx < n_joint; ++jx) {
            for (int vx = 0; vx < n_view; ++vx) {
                auto p1T = proj_mats[vx](cv::Rect{ 0, 0, 4, 1 });
                auto p2T = proj_mats[vx](cv::Rect{ 0, 1, 4, 1 });
                auto p3T = proj_mats[vx](cv::Rect{ 0, 2, 4, 1 });

                constraint_mat(cv::Rect{ 0, 2 * vx, 4, 1 }) =
                        p1T - p3T * points[vx][jx].x;
                constraint_mat(cv::Rect{ 0, 2 * vx + 1, 4, 1 }) =
                        p2T - p3T * points[vx][jx].y;
            }
            cv::SVD svd_A(constraint_mat);
            cv::Mat x_min = svd_A.vt.row(svd_A.vt.rows - 1);
            points3d.emplace_back(
                    x_min.at<double>(0, 0),
                    x_min.at<double>(0, 1),
                    x_min.at<double>(0, 2));
            points3d[jx] /= x_min.at<double>(0, 3);
        }

        return points3d;
    }

    Int_Single3dHPE::Int_Single3dHPE(Int_Single2dHPE *hpe_2d) : hpe_2d(hpe_2d) {
        assert(hpe_2d != nullptr);
    }
}
