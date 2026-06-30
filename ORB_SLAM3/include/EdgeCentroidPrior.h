#pragma once
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include <Eigen/Core>

namespace ORB_SLAM3 {

// Soft anchor: constrains a VertexSBAPointXYZ (object centroid) to stay near
// the depth-sensor-derived initial estimate. Prevents depth drift when the
// centroid vertex has only one 2D projection edge (which leaves depth unobservable).
class EdgeCentroidPrior : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeCentroidPrior() {}

    void computeError() override {
        const g2o::VertexSBAPointXYZ* v = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        _error = v->estimate() - _measurement;
    }

    bool read(std::istream&) override { return true; }
    bool write(std::ostream&) const override { return true; }
};

} // namespace ORB_SLAM3
