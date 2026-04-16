#include "VertexObject.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"
#include <Eigen/Core>

namespace ORB_SLAM3 {

VertexObject::VertexObject() {}

void VertexObject::setToOriginImpl()  {
    _estimate = Sophus::SE3d();
}

void VertexObject::oplusImpl(const double* update)  {
    Eigen::Matrix<double,6,1> upd;
    upd << update[0], update[1], update[2],
           update[3], update[4], update[5];

    _estimate = Sophus::SE3d::exp(upd) * _estimate;
}

bool VertexObject::read(std::istream& is)
{
    return false;
}

bool VertexObject::write(std::ostream& os) const
{
    return false;
}
}