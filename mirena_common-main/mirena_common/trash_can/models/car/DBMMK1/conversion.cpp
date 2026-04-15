#include "conversion.hpp"
 #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

mirena_common::msg::Car::SharedPtr DBMMK1::state_to_msg(X state)
{
    auto msg = std::make_shared<mirena_common::msg::Car>();
    StateAcessor predicted_state(state);

    msg->pose.position.set__x(predicted_state.p_x());
    msg->pose.position.set__y(predicted_state.p_y());
    msg->pose.position.set__z(0);
    tf2::Quaternion q;
    q.setRPY(0, 0, predicted_state.phi());  // RPY in radians
    geometry_msgs::msg::Quaternion q_msg;
    tf2::convert(q, q_msg);
    msg->pose.set__orientation(q_msg);

    msg->velocity.linear.set__x(cos(predicted_state.phi())*predicted_state.u()+sin(predicted_state.phi())*predicted_state.v());
    msg->velocity.linear.set__y(-sin(predicted_state.phi())*predicted_state.u()+cos(predicted_state.phi())*predicted_state.v());
    msg->velocity.linear.set__z(0);
    msg->velocity.angular.set__x(0);
    msg->velocity.angular.set__y(0);
    msg->velocity.angular.set__z(predicted_state.omega());
     
    return msg;
}