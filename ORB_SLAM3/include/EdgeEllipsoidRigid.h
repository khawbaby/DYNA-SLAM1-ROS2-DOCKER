#pragma once
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include "VertexObject.h"
#include <Eigen/Core>

namespace ORB_SLAM3 {


class EdgeEllipsoidRigid 
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 VertexObject,
                                 VertexObject>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // parametric angles
    float theta;
    float phi;

    // axes for both frames
    Eigen::Vector3d axes_prev;
    Eigen::Vector3d axes_curr;

    Eigen::Vector3d sample(const Eigen::Vector3d& axes) const;

    virtual void computeError() override;
    virtual bool read(std::istream&) { return false; }
    virtual bool write(std::ostream&) const { return false; }
};
}