#pragma once

#define DEBUG true 

#if DEBUG
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace mirena
{
    class DebugPub {
    public:
        static DebugPub get_singleton() {return _instance;};
        static void initialize_from(rclcpp::Node::SharedPtr node);
        void publish_marker(const visualization_msgs::msg::Marker &msg);
        void publish_marker(const visualization_msgs::msg::MarkerArray &msg);
    private:
        static DebugPub _instance;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _debug_marker_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _debug_marker_array_pub;
    };
}
#endif