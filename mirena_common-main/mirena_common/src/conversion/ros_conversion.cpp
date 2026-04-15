#include "ros_conversion.hpp"

using namespace mirena;

mirena_common::msg::BezierCurve mirena::to_msg(BezierCurve3D &native)
{
    mirena_common::msg::BezierCurve msg;

    int number_of_points = native.get_number_of_points();
    msg.points.resize(number_of_points);
    for (int point_index = 0; point_index < number_of_points; point_index++)
    {
        msg.points.assign(point_index, to_msg(native.get_point(point_index)));
    }
    msg.is_closed = native.is_closed();
    return msg;
}

BezierCurve3D mirena::from_msg(mirena_common::msg::BezierCurve &msg)
{

    mirena::BezierCurve3D native;

    int number_of_points = msg.points.size();
    native.reserve(number_of_points);
    for (int point_index = 0; point_index < number_of_points; point_index++)
    {
        native.insert_point(from_msg(msg.points.at(point_index)));
    }
    native.set_is_closed(msg.is_closed);
    return native;
}

mirena_common::msg::BezierCurvePoint mirena::to_msg(mirena::BezierCurve3D::BezierCurvePoint &native)
{
    mirena_common::msg::BezierCurvePoint msg;
    msg.set__point(to_msg(native.point));
    msg.set__in_control_point(to_msg(native.in_control_point));
    msg.set__out_control_point(to_msg(native.out_control_point));
    return msg;
}

mirena::BezierCurve3D::BezierCurvePoint mirena::from_msg(mirena_common::msg::BezierCurvePoint &msg)
{
    mirena::BezierCurve3D::BezierCurvePoint native;
    native.point = from_msg(msg.point);
    native.in_control_point = from_msg(msg.in_control_point);
    native.out_control_point = from_msg(msg.out_control_point);
    return native;
}

geometry_msgs::msg::Point mirena::to_msg(mirena::Point3D &native)
{
    geometry_msgs::msg::Point msg;
    msg.set__x(native.x);
    msg.set__y(native.y);
    msg.set__z(native.z);
    return msg;
}

mirena::Point3D mirena::from_msg(geometry_msgs::msg::Point &msg)
{
    mirena::Point3D native;
    native.x = msg.x;
    native.y = msg.y;
    native.z = msg.z;
    return native;
}

visualization_msgs::msg::Marker mirena::to_marker(mirena::Point3D &native, std::string ns, int id)
{
    visualization_msgs::msg::Marker marker;

    marker.ns = ns;
    marker.id = id;

    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.05;

    marker.color.a = 1.0;
    marker.color.r = 0.12;
    marker.color.g = 0.95;
    marker.color.b = 0.35;

    marker.pose.position.x = native.x;
    marker.pose.position.y = native.y;
    marker.pose.position.z = native.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    return marker;
}
