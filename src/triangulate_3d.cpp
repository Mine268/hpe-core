#include "triangulate_3d.hpp"


hpe_core::Triangulator::Triangulator(hpe_core::Int_Single2dHPE *hpe_2d) : Int_Single3dHPE(hpe_2d) {}

hpe_core::HumanPose hpe_core::Triangulator::predict(
        const vector<Mat> &multiview_img,
        const vector<Mat> &intrs,
        const vector<Mat> &rots,
        const vector<Mat> &trans
) {
    CV_Assert(multiview_img.size() == intrs.size() &&
              multiview_img.size() == rots.size() &&
              multiview_img.size() == trans.size());

    auto [multiview_pose2d, multiview_conf] = this->hpe_2d->predict(multiview_img);
    auto pose_struct = this->hpe_2d->init_struct();
    auto pose3d = triangulatePoints(multiview_pose2d, intrs, rots, trans);
    pose_struct.kps = std::move(pose3d);

    return pose_struct;
}

std::vector<hpe_core::HumanPose>
hpe_core::Triangulator::predict(const vector<vector<Mat>> &multiview_imgs,
                                const vector<vector<Mat>> &batch_intrs,
                                const vector<vector<Mat>> &batch_rots,
                                const vector<vector<Mat>> &batch_trans) {
    CV_Assert(multiview_imgs.size() == batch_intrs.size() &&
              multiview_imgs.size() == batch_rots.size() &&
              multiview_imgs.size() == batch_trans.size());

    vector<HumanPose> poses;
    for (int bx = 0; bx < multiview_imgs.size(); ++bx) {
        auto [multiview_pose2d, multiview_conf] = this->hpe_2d->predict(multiview_imgs[bx]);
        auto pose_struct = this->hpe_2d->init_struct();
        auto pose3d = triangulatePoints(multiview_pose2d, batch_intrs[bx], batch_rots[bx], batch_trans[bx]);
        pose_struct.kps = std::move(pose3d);
        poses.push_back(pose_struct);
    }

    return poses;
}


