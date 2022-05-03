#!/usr/bin/env python
from pickle import FALSE
import threading
from collections import deque

import rospy
import numpy as np
from iLQR import iLQR, Track, EllipsoidObj
from realtime_buffer import RealtimeBuffer
from traj_msgs.msg import TrajMsg
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
from scipy.spatial.transform import Rotation
from stanley_backup import Stanley
from ilqr_own import ILQR_Own
from rc_control_msgs.msg import RCControl
import yaml, csv
import time
import matplotlib.pyplot as plt

controller = "ilqr_new" # "ilqr_new, ilqr_old, or stanley"

class State():
    def __init__(self, state, t) -> None:
        self.state = state
        self.t = t
        
class Plan():
    def __init__(self, x, u, K, t0, dt, N) -> None:
        self.nominal_x = x
        self.nominal_u = u
        self.K = K
        self.t0 = t0
        self.dt = dt
        self.N = N
    
    def get_policy(self, t):
        k = int(np.floor((t-self.t0).to_sec()/self.dt))
        if k>= self.N:
            rospy.logwarn("Try to retrive policy beyond horizon")
            x_k = self.nominal_x[:,-1]
            x_k[2:] = 0
            u_k = np.zeros(2)
            K_k = np.zeros((2,4))
        else:
            x_k = self.nominal_x[:,k]
            u_k = self.nominal_u[:,k]
            K_k = self.K[:,:,k]

        return x_k, u_k, K_k

        
class Planning_MPC():

    def __init__(self,
                track_file=None,
                pose_topic='/zed2/zed_node/odom',
                lead_car_pose_topic='',
                control_topic='/planning/trajectory',
                params_file='modelparams.yaml'):
        '''
        Main class for the MPC trajectory planner
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''
        # load parameters
        rospy.loginfo(f"Params file = {params_file}")

        with open(params_file) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)

        # parameters for the ocp solver
        self.T = self.params['T']
        self.N = self.params['N']
        self.d_open_loop = np.array(self.params['d_open_loop'])
        self.replan_dt = self.T / (self.N - 1)


        # set up the optimal control solver
        if controller=="ilqr_old":
            self.ocp_solver = iLQR(params=self.params)
        elif controller=="stanley":
            gain = 1
            self.ocp_solver = Stanley(gain)
        elif controller=="ilqr_new":
            self.ocp_solver = ILRQ_Own()

        rospy.loginfo("Successfully initialized the solver with horizon " +
                    str(self.T) + "s, and " + str(self.N) + " steps.")

        self.state_buffer = RealtimeBuffer()
        self.plan_buffer = RealtimeBuffer()
        self.lead_car_state_buffer = RealtimeBuffer()
        self.lead_car_center_line = None
        self.following_car_center_line = None
        self.lead_car_traj = None

        self.i = 0

        # set up publiser to the reference trajectory and subscriber to the pose
        self.control_pub = rospy.Publisher(control_topic, RCControl, queue_size=1)

        self.pose_sub = rospy.Subscriber(pose_topic,
                                        Odometry,
                                        self.odom_sub_callback,
                                        queue_size=1)
        
        april_tag_topic = '/tag_detections'
        self.april_tag_sub = rospy.Subscriber(april_tag_topic, AprilTagDetectionArray, self.april_tag_sub_callback, queue_size=1)

        self.lead_car_pose_sub = rospy.Subscriber(lead_car_pose_topic,
                                        Odometry,
                                        self.lead_car_odom_sub_callback,
                                        queue_size=1)
        self.use_april = False
    
        # start planning thread
        # threading.Thread(target=self.ilqr_pub_thread).start()
    def april_tag_sub_callback(self, aprilMsg):

        if not self.use_april:
            return

        cur_t = aprilMsg.header.stamp
        detections = aprilMsg.detections 

        if len(detections) == 0:
            rospy.loginfo("LOST DETECTION")
            return

        detection_idx = 0
        for i, detection in enumerate(detections):
            if detection.id[0] == 573:
                detection_idx = i
        
        detection = detections[detection_idx]

        rel_x = detection.pose.pose.position.x
        rel_y = detection.pose.pose.position.y

        r = Rotation.from_quat([
            aprilMsg.pose.pose.orientation.x, aprilMsg.pose.pose.orientation.y,
            aprilMsg.pose.pose.orientation.z, aprilMsg.pose.pose.orientation.w
        ])

        rot_vec = r.as_rotvec()
        rel_psi = rot_vec[2]

        if detection.id[0] == 487:
            pass
        if detection.id[0] == 563:
            pass

        # following car state
        followcar_state = self.state_buffer.readFromRT()

        #x, y, v, psi 
        x = followcar_state.state[0] + rel_x
        y = followcar_state.state[1] + rel_y
        psi = followcar_state.state[3] + rel_psi


        # get previous state
        prev_state = self.lead_car_state_buffer.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0

        cur_X = np.array([x, y, v, psi])
        self.lead_car_state_buffer.writeFromNonRT(State(cur_X, cur_t))
        # rospy.loginfo("Leading car pose: " + str(cur_X))
        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line, np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T, axis=1) # append most recent x and y
        

    def lead_car_odom_sub_callback(self, odomMsg):
        """
        Subscriber callback function of the lead robot pose
        """
        if self.use_april:
            return
        cur_t = odomMsg.header.stamp
        # postion
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y

        # rospy.loginfo("LEAD CAR DATA: " + str(x) + " " + str(y))

        r = Rotation.from_quat([
            odomMsg.pose.pose.orientation.x, odomMsg.pose.pose.orientation.y,
            odomMsg.pose.pose.orientation.z, odomMsg.pose.pose.orientation.w
        ])

        rot_vec = r.as_rotvec()
        psi = rot_vec[2]
        
        # get previous state
        prev_state = self.lead_car_state_buffer.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0

        cur_X = np.array([x, y, v, psi])
        self.lead_car_state_buffer.writeFromNonRT(State(cur_X, cur_t))
        # rospy.loginfo("Leading car pose: " + str(cur_X))
        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T
            # self.lead_car_headings = np.array([cur_X[3]]) 
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line, np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T, axis=1) # append most recent x and y
            # self.lead_car_headings = np.append(self.lead_car_headings, [cur_X[3]], axis=0)

    def odom_sub_callback(self, odomMsg):
        """
        Subscriber callback function of the robot pose
        """

        cur_t = odomMsg.header.stamp

        # postion
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y

        # pose
        r = Rotation.from_quat([
            odomMsg.pose.pose.orientation.x, odomMsg.pose.pose.orientation.y,
            odomMsg.pose.pose.orientation.z, odomMsg.pose.pose.orientation.w
        ])

        rot_vec = r.as_rotvec()
        psi = rot_vec[2]
        
        # get previous state
        prev_state = self.state_buffer.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0
        cur_X = np.array([x, y, v, psi])
        # rospy.loginfo(cur_X)


        if self.following_car_center_line is None:
            self.following_car_center_line = np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T
        else:
            self.following_car_center_line = np.append(self.following_car_center_line, np.array([[cur_X[0], cur_X[1], cur_X[3]]]).T, axis=1) # append most recent x, y, psi


        self.ocp_solver.set_ref_path(self.lead_car_center_line)
        sol_u, target_idx = self.ocp_solver.solve(cur_X)

        # rospy.loginfo(sol_u[1])

        steer_angle = self.normalize_angle(cur_X[3] + sol_u[1])

        # rospy.loginfo(steer_angle)

        lead_car_line = self.ocp_solver.ref_path

        plt.figure()
        plt.scatter(lead_car_line[0,:], lead_car_line[1,:], color='orange')
        plt.arrow(lead_car_line[0, target_idx], lead_car_line[1, target_idx], np.cos(lead_car_line[2, target_idx]), np.sin(lead_car_line[2, target_idx]), color='black')
        plt.scatter(self.following_car_center_line[0,:], self.following_car_center_line[1,:], color= 'blue')
        plt.arrow(cur_X[0], cur_X[1], np.cos(cur_X[3]), np.sin(cur_X[3]), color = 'blue')
        plt.arrow(cur_X[0], cur_X[1], np.cos(steer_angle), np.sin(steer_angle), color= 'green')
        plt.savefig('stanley/test' + str(self.i))

        self.i += 1
        # if v < .3:
        #     a = .2
        # else: 
        #     a = 0
        self.publish_control(v, [sol_u[0], sol_u[1]], cur_t)
        
        # write the new pose to the buffer
        self.state_buffer.writeFromNonRT(State(cur_X, cur_t))

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        
        return angle

    def publish_control(self, v, u, cur_t):
        control = RCControl()
        control.header.stamp = cur_t
        a = u[0]
        
        delta = -u[1] # wtf why?
        #delta = u[1]
        
        if a<0:
            d = a/10-0.5
        else:
            temp = np.array([v**3, v**2, v, a**3, a**2, a, v**2*a, v*a**2, v*a, 1])
            d = temp@self.d_open_loop
            d = d+min(delta*delta*0.5,0.05)

        control.throttle = np.clip(d, -1.0, 0.1)
        # control.steer = np.clip(delta/.3, -.8, .8)
        control.steer = np.clip(delta/.3, -1.0, 1.0)
        control.reverse = False
        # rospy.loginfo("a: " + str(a) + " throttle: " + str(control.throttle))
        self.control_pub.publish(control)

    def run(self):
        rospy.spin() 


