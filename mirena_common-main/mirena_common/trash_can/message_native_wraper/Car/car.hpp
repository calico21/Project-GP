#ifndef car_hpp
#define car_hpp

#define ROSCar mirena_common::msg::Car
#define ROSTwist geometry_msgs::msg::Twist 

#include "rclcpp/node.hpp"
#include "Common/msg_wrapper.hpp"

#include "Entity/entity.hpp"
#include "mirena_common/msg/car.hpp"

namespace mirena
{

class Car : public MsgWrapper<ROSCar> {
    public:

    Car();
    Car(Eigen::Vector2d& position, double rotation, ROSTwist& velocity, ROSTwist& acceleration); 
    Car(ROSCar::SharedPtr msg) : MsgWrapper(msg) {}
    template <typename Owner>
    Car(std::shared_ptr<Owner> owner, ROSCar::RawPtr msg) : MsgWrapper(owner, msg) {}
    ~Car();

    double get_yaw();
    ROSTwist get_speed();
    ROSTwist get_acceleration();
    Entity get_entity();
    
    // Unwrap and copy the message
    ROSCar::SharedPtr to_msg() const;

    // Unwrap, copy & timestamp the message
    ROSCar::SharedPtr to_msg(rclcpp::Node& creator_node, std::string frame_id) const;

    static mirena_common::msg::Car build_msg(Eigen::Vector2d& position, double rotation, ROSTwist& velocity, ROSTwist& acceleration);

};
    
} // namespace mirena

#endif