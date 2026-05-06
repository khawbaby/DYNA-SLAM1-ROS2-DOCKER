#include "EdgeEllipsoidRigid.h"
#include "VertexObject.h"

namespace ORB_SLAM3 {
Eigen::Vector3d EdgeEllipsoidRigid::sample(
    const Eigen::Vector3d& a) const
{
    double x = sin(theta) * cos(phi);
    double y = sin(theta) * sin(phi);
    double z = cos(theta);

    return Eigen::Vector3d(
        a[0] * x,
        a[1] * y,
        a[2] * z
    );
}

void EdgeEllipsoidRigid::computeError()
{
    const auto* vPrev = static_cast<const VertexObject*>(_vertices[0]);
    const auto* vCurr = static_cast<const VertexObject*>(_vertices[1]);

    Sophus::SE3d T_prev = vPrev->estimate();
    Sophus::SE3d T_curr = vCurr->estimate();

    // sample corresponding parametric points
    Eigen::Vector3d X_prev = sample(axes_prev);
    Eigen::Vector3d X_curr = sample(axes_curr);

    Eigen::Vector3d Pw_prev = T_prev * X_prev;
    Eigen::Vector3d Pw_curr = T_curr * X_curr;

    if(!Pw_prev.allFinite() || !Pw_curr.allFinite())
    {
        _error.setZero();
        return;
    }

    _error = Pw_curr - Pw_prev;
}
}