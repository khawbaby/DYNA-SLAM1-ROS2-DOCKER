#include "EdgeCameraObject.h"
#include "Thirdparty/Sophus/sophus/se3.hpp"

namespace ORB_SLAM3 {

EdgeCameraObject::EdgeCameraObject()
{
    pCamera = nullptr;
}

void EdgeCameraObject::computeError()
{
    const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const auto* vObj = static_cast<const VertexObject*>(_vertices[1]);

    // --- Camera pose ---
    g2o::SE3Quat Tcw_g2o = vCam->estimate();
    Sophus::SE3d Tcw(Tcw_g2o.rotation(), Tcw_g2o.translation());

    // --- Object pose ---
    Sophus::SE3d Two = vObj->estimate();

    // --- Transform object point to camera ---
    Eigen::Vector3d Xc = Tcw * Two * X_obj;

    // --- Projection ---
    Eigen::Vector2d proj = pCamera->project(Xc);

    // --- Error ---
    _error = _measurement - proj;
}

void EdgeCameraObject::linearizeOplus()
{
    // For now: let g2o use numeric Jacobians
    // (this is fine for initial testing)

    g2o::BaseBinaryEdge<
        2, Eigen::Vector2d,
        g2o::VertexSE3Expmap,
        VertexObject
    >::linearizeOplus();
}

bool EdgeCameraObject::read(std::istream& is)
{
    return false;
}

bool EdgeCameraObject::write(std::ostream& os) const
{
    return false;
}

}