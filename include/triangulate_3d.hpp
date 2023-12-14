#pragma once


#include "hpe_core.hpp"

namespace hpe_core {

    class Triangulator : public virtual Int_Single3dHPE {
    public:
        explicit Triangulator(Int_Single2dHPE *hpe_2d);

        HumanPose predict(
                const vector<Mat> &multiview_img,
                const vector<Mat> &intrs,
                const vector<Mat> &rots,
                const vector<Mat> &trans
        ) override;

        vector<HumanPose> predict(
                const vector<vector<Mat>> &multiview_imgs,
                const vector<vector<Mat>> &batch_intrs,
                const vector<vector<Mat>> &batch_rots,
                const vector<vector<Mat>> &batch_trans
        ) override;
    };

}
