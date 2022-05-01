#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from planning import Planning_MPC
import sys, os


def main():
    rospy.init_node('traj_planning_node')
    rospy.loginfo("Start trajectory planning node")
    ## read parameters
    ControllerTopic = rospy.get_param("~ControllerTopic")
    LeadCarPoseTopic = rospy.get_param("~LeadCarPoseTopic")
    PoseTopic = rospy.get_param("~PoseTopic")
    ParamsFile = rospy.get_param("~PlanParamsFile")
    TrackFile = rospy.get_param("~TrackFile")    

    planner = Planning_MPC(track_file=TrackFile,
                 pose_topic=PoseTopic,
                 lead_car_pose_topic=LeadCarPoseTopic,
                 control_topic=ControllerTopic,
                 params_file=ParamsFile)
    planner.run()


if __name__ == '__main__':
    main()
