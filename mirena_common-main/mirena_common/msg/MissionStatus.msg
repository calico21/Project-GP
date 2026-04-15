std_msgs/Header header

# Mission Status Constants
uint8 IDLE = 0
uint8 CONFIGURING = 1
uint8 MISSION_ONGOING = 2
uint8 MISSION_FINISHED = 3
uint8 MISSION_FAILED = 4

# The actual status field
uint8 state

# Mission data
uint32 lap_counter
uint32 cones_count_actual
uint32 cones_count_total