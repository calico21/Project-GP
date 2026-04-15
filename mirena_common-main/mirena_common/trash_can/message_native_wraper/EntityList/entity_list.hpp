#ifndef entity_list_hpp
#define entity_list_hpp

#define ROSEntityList mirena_common::msg::EntityList

#include <vector>

#include "mirena_common/msg/entity_list.hpp"
#include "rclcpp/node.hpp"

#include "Entity/entity.hpp"
#include "Common/msg_wrapper.hpp"

namespace mirena {

// Adapter class to act as a standardized wrapper over EntityList.msg
class EntityList : public MsgWrapper<ROSEntityList>{
    public:
    class Iterator {
        std::shared_ptr<ROSEntityList> _owner;
        std::vector<ROSEntity, std::allocator<ROSEntity>>::iterator _it;
    
    public:
        explicit Iterator(std::shared_ptr<ROSEntityList> owner, std::vector<ROSEntity, std::allocator<ROSEntity>>::iterator ptr) : _owner(owner), _it(ptr) {}
    
        Entity operator*(); // Wrap the entity msg
        Iterator operator++(); // Forward the advacing of the iterator
        bool operator!=(const Iterator &other) const; // Forward the vectors iterator
    };

    EntityList();
    EntityList(std::vector<Entity>& entities);
    EntityList(ROSEntityList::SharedPtr msg) : MsgWrapper(msg){}
    template <typename Owner>
    EntityList(std::shared_ptr<Owner> owner, ROSEntity::RawPtr msg) : MsgWrapper(owner, msg) {}
    ~EntityList() {};

    Entity at(int index);

    int size();
    
    EntityList::Iterator begin();
    EntityList::Iterator end();

    void push_back(const ROSEntity& entity);

    // Unwrap and copy the message
    ROSEntityList::SharedPtr to_msg() const;

    // Unwrap, copy & timestamp the message
    ROSEntityList::SharedPtr to_msg(rclcpp::Node& creator_node, std::string frame_id) const;


    static ROSEntityList build_msg(std::vector<Entity>& entities);

};

}

#endif