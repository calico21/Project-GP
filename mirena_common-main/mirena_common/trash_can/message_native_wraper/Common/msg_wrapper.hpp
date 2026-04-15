#ifndef msg_wrapper_hpp
#define msg_wrapper_hpp

#include <memory>

namespace mirena {

// Template parent class for message wrappers
// May act as propper wrapper, owning the message, or act as an view acting over a reference to a message owned elsewhere
template <typename MSG>
class MsgWrapper
{
    public:
    // Safely Borrow the message
    template <typename Owner>
    MsgWrapper(std::shared_ptr<Owner> owner, MSG* msg) : _msg(std::shared_ptr<MSG>(owner, msg)) {}

    // Own the message
    MsgWrapper(std::shared_ptr<MSG> msg) : _msg(msg) {};

    // Return a new message (copy)
    std::shared_ptr<MSG> to_msg() const { return std::make_shared<MSG>(*this->_msg); };

    // Return reference to held message 
    std::shared_ptr<MSG> get_msg() const {return this->_msg;}

    private:
    std::shared_ptr<MSG> _msg; 
};

}

#endif