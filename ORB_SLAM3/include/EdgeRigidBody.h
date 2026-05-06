#pragma once
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include "VertexObject.h"
#include <Eigen/Core>

namespace ORB_SLAM3 {

class EdgeRigidBody 
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 VertexObject,
                                 VertexObject>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // feature point in object frame at t-1
    Eigen::Vector3d X_prev;

    // feature point in object frame at t
    Eigen::Vector3d X_curr;

    virtual void computeError() override;

    virtual bool read(std::istream&) { return false; }
    virtual bool write(std::ostream&) const { return false; }
};
}