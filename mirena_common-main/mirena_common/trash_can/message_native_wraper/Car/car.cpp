#include "car.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>

using namespace mirena;

Car::Car() : 
MsgWrapper(std::make_shared<ROSCar>()) 
{}

Car::Car(Eigen::Vector2d& position, double rotation, ROSTwist& velocity, ROSTwist& acceleration) : 
Car(std::make_shared<ROSCar>(Car::build_msg(position, rotation, velocity, acceleration))) 
{};

Car::~Car() {}

// In yaw returned in radians
double mirena::Car::get_yaw()
{
    // Convert to a rotation matrix
    tf2::Quaternion quaternion_tf;
    tf2::fromMsg(this->get_msg()->orientation, quaternion_tf);
    quaternion_tf.normalize();
    tf2::Matrix3x3 matrix(quaternion_tf);
    
    // Extract RPY angles from the matrix
    double roll, pitch, yaw;
    matrix.getRPY(roll, pitch, yaw);
    return yaw;
}

ROSTwist mirena::Car::get_speed()
{
    return this->get_msg()->velocity;
}

ROSTwist mirena::Car::get_acceleration()
{
    return this->get_msg()->acceleration;
}

Entity mirena::Car::get_entity()
{
    return Entity(this->get_msg(), &this->get_msg()->entity);
}

ROSCar::SharedPtr mirena::Car::to_msg() const
{
    return this->MsgWrapper::to_msg(); 
}

ROSCar::SharedPtr mirena::Car::to_msg(rclcpp::Node &creator_node, std::string frame_id) const
{
    ROSCar::SharedPtr msg = this->to_msg();
    msg->header.set__frame_id(frame_id);
    msg->header.set__stamp(creator_node.get_clock()->now());
    return msg;
}

ROSCar mirena::Car::build_msg(Eigen::Vector2d &position, double rotation, ROSTwist &velocity, ROSTwist &acceleration)
{
    ROSCar new_msg;
    new_msg.set__entity(Entity::build_msg(position, "Car", 1.0));
    tf2::Quaternion pose;
    pose.setRPY(rotation, 0, 0);
    new_msg.set__orientation(tf2::toMsg(pose));
    new_msg.set__velocity(velocity);
    new_msg.set__acceleration(acceleration);
    return new_msg;
};
