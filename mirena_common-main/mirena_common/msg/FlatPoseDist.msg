# This message conveys a 2d (Flat) Pose expressed as a multivariate normal distribution given by its mean and covariance
#
# The mean vector is structured as follows (x_pos [m], y_pos [m], yaw [rad])
# The convariance matrix follows from the structure of the mean vector, with element (i, j) of the covariance matrix corresponding to index (i * 3 + j) of the covariance in the message#
#

std_msgs/Header header

float64[3] mean
float64[9] covariance 

