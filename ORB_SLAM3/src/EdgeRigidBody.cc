#include "EdgeRigidBody.h"
#include "VertexObject.h"

namespace ORB_SLAM3 {
void EdgeRigidBody::computeError()
{
    const auto* vPrev = static_cast<const VertexObject*>(_vertices[0]);
    const auto* vCurr = static_cast<const VertexObject*>(_vertices[1]);

    Sophus::SE3d T_prev = vPrev->estimate();
    Sophus::SE3d T_curr = vCurr->estimate();

    // transform both to world
    Eigen::Vector3d Pw_prev = T_prev * X_prev;
    Eigen::Vector3d Pw_curr = T_curr * X_curr;

    if(!Pw_prev.allFinite() || !Pw_curr.allFinite())
    {
        _error.setZero();
        return;
    }

    // error = difference in world
    _error = Pw_curr - Pw_prev;
}
}