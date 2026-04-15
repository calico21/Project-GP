#include "debug.hpp"
#include <iostream>

using namespace mirena;

#if DEBUG
DebugPub DebugPub::_instance;

void mirena::DebugPub::initialize_from(rclcpp::Node::SharedPtr node)
{
        std::string debug_topic_base_name = "/debug/";
        _instance._debug_marker_pub = node->create_publisher<visualization_msgs::msg::Marker>(debug_topic_base_name+node->get_name()+"/marker", 10);
        _instance._debug_marker_array_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>(debug_topic_base_name+node->get_name()+"/marker_array", 10);
}

void mirena::DebugPub::publish_marker(const visualization_msgs::msg::Marker &msg)
{
        if(_debug_marker_pub == nullptr){
                std::cerr << "mirena::DebugPub::initialize_from was never called, so no debug messages can be published" << std::endl;
                return;
        }
        _debug_marker_pub->publish(msg);
}

void mirena::DebugPub::publish_marker(const visualization_msgs::msg::MarkerArray &msg)
{
        if(_debug_marker_array_pub == nullptr){
                std::cerr << "mirena::DebugPub::initialize_from was never called, so no debug messages can be published" << std::endl;
                return;
        }
        _debug_marker_array_pub->publish(msg);
}

#endif
