#! /bin/sh
HOST_IP=192.168.0.113
CLIENT_IP=192.168.0.104
export ROS_IP=$CLIENT_IP
export ROS_MASTER_URI=http://$HOST_IP:11311
export ROS_HOSTNAME=$CLIENT_IP