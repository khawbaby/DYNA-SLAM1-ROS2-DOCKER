#pragma once

#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "VertexObject.h"
#include "GeometricCamera.h"   // or GeometricCamera depending on your setup

#include <Eigen/Core>

namespace ORB_SLAM3 {

class EdgeCameraObject :
    public g2o::BaseBinaryEdge<
        2,                      // error dimension (u,v)
        Eigen::Vector2d,        // measurement
        g2o::VertexSE3Expmap,   // camera pose
        VertexObject            // object pose
    >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeCameraObject();

    // 3D point in object frame
    Eigen::Vector3d X_obj;

    // camera model (for computeError)
    GeometricCamera* pCamera;

    // intrinsics for analytic Jacobian (pinhole)
    double fx = 0, fy = 0;

    // core functions
    void computeError() override;
    void linearizeOplus() override;

    bool read(std::istream& is) override;
    bool write(std::ostream& os) const override;
};

}