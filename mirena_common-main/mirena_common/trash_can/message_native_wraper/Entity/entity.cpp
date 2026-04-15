#include "entity.hpp"

using namespace mirena;


Entity::Entity() :
MsgWrapper(std::make_shared<ROSEntity>()) {}

Entity::Entity(double pos_x, double pos_y) : Entity(Eigen::Vector2d(pos_x, pos_y), "", 1.0) {};

Entity::Entity(const Eigen::Vector2d& position, std::string type, double confidence) : 
MsgWrapper(std::make_shared<ROSEntity>(Entity::build_msg(position, type, confidence))){};

Entity::~Entity() {}

Eigen::Vector2d Entity::get_position() const
{
    return Eigen::Vector2d(this->get_msg()->position.x, this->get_msg()->position.y);
}

std::string Entity::get_type() const
{
    return this->get_msg()->type;
}

double Entity::get_confidence() const 
{
    return this->get_msg()->confidence;
}

double Entity::get_x() const { return this->get_position().x(); }

double Entity::get_y() const { return this->get_position().y(); }

void Entity::set_position(Eigen::Vector2d &position)
{
    this->get_msg()->position.set__x(position.x());
    this->get_msg()->position.set__y(position.y());
}

void Entity::set_type(std::string type)
{
    this->get_msg()->set__type(type);
}

void Entity::set_confidence(double confidence)
{
    this->get_msg()->set__confidence(confidence);
}

ROSEntity Entity::build_msg(const Eigen::Vector2d &position, std::string type, double confidence)
{
    ROSEntity new_msg;
    new_msg.position.set__x(position.x());
    new_msg.position.set__y(position.y());
    new_msg.set__type(type);
    new_msg.set__confidence(confidence);
    return new_msg;
}