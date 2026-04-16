#include "EdgeObjectMotion.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include "VertexObject.h"
#include <Eigen/Core>

namespace ORB_SLAM3 {
EdgeObjectMotion::EdgeObjectMotion()
{
}

void EdgeObjectMotion::computeError()
{
    const VertexObject* vPrev =
        static_cast<const VertexObject*>(_vertices[0]);

    const VertexObject* vCurr =
        static_cast<const VertexObject*>(_vertices[1]);

    Sophus::SE3d T_prev = vPrev->estimate();
    Sophus::SE3d T_curr = vCurr->estimate();

    // Predicted motion
    Sophus::SE3d T_pred = T_prev.inverse() * T_curr;

    // Error = measurement^{-1} * prediction
    Sophus::SE3d T_err = _measurement.inverse() * T_pred;

    _error = T_err.log();   // 6D vector
}

bool EdgeObjectMotion::read(std::istream& is)
{
    return false;
}

bool EdgeObjectMotion::write(std::ostream& os) const
{
    return false;
}
}