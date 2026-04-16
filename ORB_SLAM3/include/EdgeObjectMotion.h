#pragma once
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include "VertexObject.h"
#include <Eigen/Core>

namespace ORB_SLAM3 {
class EdgeObjectMotion :
    public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexObject, VertexObject>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeObjectMotion();

    void computeError() override;

    virtual bool read(std::istream& is) override;
    virtual bool write(std::ostream& os) const override;
};
}