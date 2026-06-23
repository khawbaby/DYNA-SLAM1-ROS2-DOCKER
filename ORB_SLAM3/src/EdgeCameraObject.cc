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
    // Analytic Jacobians for e = z - proj(Tcw * Two * X_obj)
    //
    // Vertex 0: camera pose (VertexSE3Expmap, left perturbation [ω; v])
    //   ∂e/∂[ω;v] = [ J_proj * skew(Xc)  |  -J_proj ]
    //
    // Vertex 1: object pose (VertexObject, left perturbation [ω; v])
    //   ∂e/∂[ω;v] = [ (J_proj*R_cw) * skew(Xw)  |  -(J_proj*R_cw) ]
    //
    // where J_proj = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]

    if(fx == 0 || fy == 0)
    {
        // Fall back to numeric if intrinsics not set
        g2o::BaseBinaryEdge<2, Eigen::Vector2d,
            g2o::VertexSE3Expmap, VertexObject>::linearizeOplus();
        return;
    }

    const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const auto* vObj = static_cast<const VertexObject*>(_vertices[1]);

    g2o::SE3Quat Tcw_g2o = vCam->estimate();
    Eigen::Matrix3d R_cw = Tcw_g2o.rotation().toRotationMatrix();
    Eigen::Vector3d t_cw = Tcw_g2o.translation();

    Sophus::SE3d Two = vObj->estimate();
    Eigen::Vector3d Xw = Two * X_obj;
    Eigen::Vector3d Xc = R_cw * Xw + t_cw;

    if(Xc.z() < 1e-6)
    {
        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();
        return;
    }

    double x = Xc.x(), y = Xc.y(), z = Xc.z();
    double invz  = 1.0 / z;
    double invz2 = invz * invz;

    // ∂proj/∂Xc  (2×3, pinhole)
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx*invz,       0, -fx*x*invz2,
                    0, fy*invz, -fy*y*invz2;

    // skew(Xc) for camera Jacobian
    Eigen::Matrix3d skew_Xc;
    skew_Xc <<  0, -z,  y,
                z,  0, -x,
               -y,  x,  0;

    // Camera pose Jacobian (vertex 0)
    _jacobianOplusXi << J_proj * skew_Xc, -J_proj;

    // skew(Xw) for object Jacobian
    double xw = Xw.x(), yw = Xw.y(), zw = Xw.z();
    Eigen::Matrix3d skew_Xw;
    skew_Xw <<   0, -zw,  yw,
               zw,    0, -xw,
              -yw,   xw,   0;

    // Object pose Jacobian (vertex 1)
    Eigen::Matrix<double,2,3> A = J_proj * R_cw;
    _jacobianOplusXj << A * skew_Xw, -A;
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