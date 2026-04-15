#ifndef entity_hpp
#define entity_hpp

#define ROSEntity mirena_common::msg::Entity

#include "Common/msg_wrapper.hpp"

#include <eigen3/Eigen/Eigen>
#include "mirena_common/msg/entity.hpp"

namespace mirena {

// Adapter class to act as a standardized wrapper over Entity.msg
class Entity : public MsgWrapper<ROSEntity>{
    public:

    Entity();
    Entity(double pos_x, double pos_y);
    Entity(const Eigen::Vector2d& position, std::string type, double confidence);
    Entity(ROSEntity::SharedPtr msg) : MsgWrapper(msg) {} 
    template <typename Owner>
    Entity(std::shared_ptr<Owner> owner, ROSEntity::RawPtr msg) : MsgWrapper(owner, msg) {}
    ~Entity();

    Eigen::Vector2d get_position() const;
    std::string get_type() const;
    double get_confidence() const;
    double get_x() const; 
    double get_y() const;

    void set_position(Eigen::Vector2d& position);
    void set_type(std::string type);
    void set_confidence(double confidence);

    static mirena_common::msg::Entity build_msg(const Eigen::Vector2d& position, std::string type, double confidence);

};

}

#endif