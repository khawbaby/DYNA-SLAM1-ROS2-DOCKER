#pragma once
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include <Eigen/Core>

namespace ORB_SLAM3 {
    class VertexObject : public g2o::BaseVertex<6, Sophus::SE3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexObject();

        virtual void setToOriginImpl() override;

        virtual void oplusImpl(const double* update) override;

        virtual bool read(std::istream&) override;
        virtual bool write(std::ostream&) const override;
    };

}