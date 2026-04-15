#include "entity_list.hpp"

using namespace mirena;

// ENTITYLIST::ITERATOR
Entity EntityList::Iterator::operator*() { return Entity(_owner, &(*_it)); } // Wrap the entity msg
EntityList::Iterator EntityList::Iterator::operator++(){ ++_it; return *this; } // Forward the advacing of the iterator
bool EntityList::Iterator::operator!=(const Iterator &other) const { return _it != other._it; } // Forward the vectors iterator


// ENTITYLIST

EntityList::EntityList() : MsgWrapper(std::make_shared<ROSEntityList>())
{
}

EntityList::EntityList(std::vector<Entity> &entities) : EntityList(std::make_shared<ROSEntityList>(EntityList::build_msg(entities)))
{
}

Entity mirena::EntityList::at(int index)
{
    return Entity(this->get_msg(), &(this->get_msg()->entities.at(index)));
}

void EntityList::push_back(const ROSEntity &entity)
{
    this->get_msg()->entities.push_back(entity);
}

int mirena::EntityList::size()
{
    return this->get_msg()->entities.size();
}

EntityList::Iterator mirena::EntityList::begin()
{
    return Iterator(this->get_msg(), this->get_msg()->entities.begin());
}

EntityList::Iterator mirena::EntityList::end()
{
    return Iterator(this->get_msg(), this->get_msg()->entities.end());
}

ROSEntityList::SharedPtr mirena::EntityList::to_msg() const
{
    return this->MsgWrapper::to_msg(); 
}

ROSEntityList::SharedPtr mirena::EntityList::to_msg(rclcpp::Node &creator_node, std::string frame_id) const
{
    ROSEntityList::SharedPtr msg = this->to_msg();
    msg->header.set__frame_id(frame_id);
    msg->header.set__stamp(creator_node.get_clock()->now());
    return msg;
}

ROSEntityList EntityList::build_msg(std::vector<Entity> &entities)
{
    ROSEntityList new_msg;
    for (auto &entity : entities)
    {
        new_msg.entities.push_back(*entity.to_msg());
    }
    return new_msg;
}
