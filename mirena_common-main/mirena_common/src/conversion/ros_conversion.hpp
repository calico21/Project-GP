#pragma once

#include "visualization_msgs/msg/marker.hpp"
#include "common_lib/bezier_curve/bezier_curve.hpp"
#include "mirena_common/msg/bezier_curve.hpp"

namespace mirena{
    
    // BezierCurve3D
    mirena_common::msg::BezierCurve to_msg(mirena::BezierCurve3D &native);
    mirena::BezierCurve3D from_msg(mirena_common::msg::BezierCurve& msg);
    mirena_common::msg::BezierCurvePoint to_msg(mirena::BezierCurve3D::BezierCurvePoint &native);
    mirena::BezierCurve3D::BezierCurvePoint from_msg(mirena_common::msg::BezierCurvePoint& msg);

    // Point3D
    geometry_msgs::msg::Point to_msg(mirena::Point3D &native);
    mirena::Point3D from_msg(geometry_msgs::msg::Point& msg);
    visualization_msgs::msg::Marker to_marker(mirena::Point3D &native, std::string ns, int id);
}