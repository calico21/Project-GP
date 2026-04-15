std_msgs/Header header

# Autonomous System States (Standard FSD Figure 15)
uint8 AS_OFF = 0
uint8 AS_READY = 1
uint8 AS_DRIVING = 2
uint8 AS_FINISHED = 3
uint8 AS_EMERGENCY = 4

uint8 state

# Autonomous Mission Types
uint8 MISSION_NONE = 0
uint8 ACCELERATION = 1
uint8 SKIDPAD = 2
uint8 AUTOCROSS = 3
uint8 TRACKDRIVE = 4
uint8 BRAKE_TEST = 5
uint8 INSPECTION = 6

uint8 mission