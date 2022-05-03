#!/usr/bin/env python
from pickle import FALSE
from telnetlib import NOP
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
#from ilqr_own import ILQR_Own
from rc_control_msgs.msg import RCControl
import yaml, csv
import time
import matplotlib.pyplot as plt
from overtaker import Overtaker

controller = "ilqr_old" # "ilqr_new, ilqr_old, or stanley"

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

        
class Planning_MPC_Overtake():

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
            randx, randy = [0,1,2,3,4,5,6], [0,1,2,3,4,5,6]
            random_track =  Track(center_line=np.array([randx,randy]),
                                width_left=self.params['track_width_L'],
                                width_right=self.params['track_width_R'],
                                loop=True)

            self.ocp_solver = iLQR(params=self.params, ref_path=random_track)
        elif controller=="stanley":
            gain = 1
            self.ocp_solver = Stanley(gain)
        #elif controller=="ilqr_new":
            #self.ocp_solver = ILRQ_Own()

        rospy.loginfo("Successfully initialized the solver with horizon " +
                    str(self.T) + "s, and " + str(self.N) + " steps.")

        self.state_buffer = RealtimeBuffer()
        self.plan_buffer = RealtimeBuffer()
        self.lead_car_state_buffer_true = RealtimeBuffer()
        self.lead_car_state_buffer_april = RealtimeBuffer()
        self.lead_car_center_line = None
        self.lead_car_center_line_true = None
        self.following_car_center_line = None
        self.lead_car_traj = None
        self.lead_car_traj_raw = None

        self.i = 0

        # set up publiser to the reference trajectory and subscriber to the pose
        self.control_pub = rospy.Publisher(control_topic, RCControl, queue_size=1)

        self.pose_sub = rospy.Subscriber(pose_topic,
                                        Odometry,
                                        self.odom_sub_callback,
                                        queue_size=1)
        
        april_tag_topic = '/nx4/tag_detections'
        self.april_tag_sub = rospy.Subscriber(april_tag_topic, AprilTagDetectionArray, self.april_tag_sub_callback, queue_size=1)

        self.lead_car_pose_sub = rospy.Subscriber(lead_car_pose_topic,
                                        Odometry,
                                        self.lead_car_odom_sub_callback,
                                        queue_size=1)
        self.use_april = False

        self.stopping = False

        self.overtaker = Overtaker()
    
        # start planning thread
        threading.Thread(target=self.ilqr_pub_thread).start()
    
    def april_tag_sub_callback(self, aprilMsg):
        
        if not self.use_april:
            return

        cur_t = aprilMsg.header.stamp
        detections = aprilMsg.detections 
        
        if len(detections) == 0:
            rospy.loginfo("LOST DETECTION")
            return
        #else:
            #rospy.loginfo("Num detections: " + str(len(detections)))

        detection_idx = 0
        for i, detection in enumerate(detections):
            if detection.id[0] == 500:
                detection_idx = i
        
        detection = detections[detection_idx]

        rel_x = detection.pose.pose.pose.position.x
        rel_y = detection.pose.pose.pose.position.y

        r = Rotation.from_quat([
            detection.pose.pose.pose.orientation.x, detection.pose.pose.pose.orientation.y,
            detection.pose.pose.pose.orientation.z, detection.pose.pose.pose.orientation.w
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
        prev_state = self.lead_car_state_buffer_april.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0

        cur_X = np.array([x, y, v, psi])
        rospy.loginfo("Lead car state (april): " + str(cur_X))
        self.lead_car_state_buffer_april.writeFromNonRT(State(cur_X, cur_t))
        # rospy.loginfo("Leading car pose: " + str(cur_X))
        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1]]]).T
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line, np.array([[cur_X[0], cur_X[1]]]).T, axis=1) # append most recent x and y

        if self.lead_car_center_line.shape[1] > self.N + 17:
            rospy.loginfo("Following")
            nominal_traj = self.lead_car_center_line[:, -10-self.N:-10]
            self.lead_car_traj_raw = nominal_traj
        

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
        prev_state = self.lead_car_state_buffer_true.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0

        cur_X = np.array([x, y, v, psi])
        rospy.loginfo("Lead car state (true): " + str(cur_X))
        self.lead_car_state_buffer_true.writeFromNonRT(State(cur_X, cur_t))
        

        # if self.lead_car_center_line_true is None:
        #     self.lead_car_center_line_true = np.array([[cur_X[0], cur_X[1]]]).T
        # else:
        #     self.lead_car_center_line_true = np.append(self.lead_car_center_line_true, np.array([[cur_X[0], cur_X[1]]]).T, axis=1)
        # return

        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1]]]).T
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line, np.array([[cur_X[0], cur_X[1]]]).T, axis=1) # append most recent x and y
                
        # rospy.loginfo(self.lead_car_center_line.shape)

        followcar_state = self.state_buffer.readFromRT()
        if followcar_state is None:
            return
        overtake_state = self.overtaker.pass_car(followcar_state.state, cur_X)
        if overtake_state is not None:
            rospy.loginfo("OVERTAKE IN PROGRESS")
            nominal_traj = np.repeat(overtake_state[:2, np.newaxis], self.N, axis=1)
            self.lead_car_traj_raw = nominal_traj
        elif self.lead_car_center_line.shape[1] > self.N + 17:
            rospy.loginfo("Following")
            # create nominal trajectory not counting last 10 to keep following vehicle behind (max len of nominal traj is 300)
            # if self.lead_car_center_line.shape[1] > 345:
            #     nominal_traj = self.lead_car_center_line[:, -300:-30]
            # else:
            #     nominal_traj = self.lead_car_center_line[:,:-30]
            
            nominal_traj = self.lead_car_center_line[:, -10-self.N:-10]
            #nominal_traj += np.random.rand(nominal_traj.shape)
            self.lead_car_traj_raw = nominal_traj

            # self.lead_car_traj = Track(center_line=nominal_traj,
            #                     width_left=self.params['track_width_L'],
            #                     width_right=self.params['track_width_R'],
            #                     loop=True)

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

        last_plan = self.plan_buffer.readFromRT()
        if last_plan is not None:
            # get the control policy
            X_k, u_k, K_k = last_plan.get_policy(cur_t)
            u = u_k+ K_k@(cur_X - X_k)        

            # if self.use_april:
            #     lead_car_state = self.lead_car_state_buffer_april.readFromRT()
            # else:
            #     lead_car_state = self.lead_car_state_buffer_true.readFromRT()
           


            # rospy.loginfo("distance: " + str(np.linalg.norm(lead_car_state.state[:2] - cur_X[:2])))
            # rospy.loginfo("velocity: " + str(lead_car_state.state[2]))
            # rospy.loginfo("----")

            # if lead_car_state.state[2] <= 0.05 and np.linalg.norm(lead_car_state.state[:2] - cur_X[:2]) <= .5:
            #     rospy.loginfo("TOO CLOSE")
            #     control = RCControl()
            #     if self.stopping is False:
            #         control.throttle = -1
            #     else:
            #         control.throttle = 0
            #     self.stopping = True
            #     control.steer = 0
            #     control.reverse = False
            #     self.control_pub.publish(control)
            #     return
            
            self.stopping = False
            rospy.loginfo("publishing control")   
            self.publish_control(v, u, cur_t)

            self.i += 1
            # plt.figure()
            # plt.scatter(self.lead_car_center_line[0,:], self.lead_car_center_line[1,:], color='orange')
            # plt.scatter(self.following_car_center_line[0,:], self.following_car_center_line[1,:], color= 'blue')
            # plt.scatter(self.lead_car_center_line_true[0,:], self.lead_car_center_line_true[1,:], color='red')
            # plt.plot(last_plan.nominal_x[0,:], last_plan.nominal_x[1,:], color='green')
            # plt.savefig('ilqr/test' + str(self.i))
            # plt.close()
            
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

        control.throttle = np.clip(d, -1.0, 1.0)
        # control.steer = np.clip(delta/.3, -.8, .8)
        control.steer = np.clip(delta/.3, -1.0, 1.0)
        control.reverse = False
        # rospy.loginfo("a: " + str(a) + " throttle: " + str(control.throttle))
        self.control_pub.publish(control)

    def ilqr_pub_thread(self):
        time.sleep(5)
        rospy.loginfo("iLQR Planning publishing thread started")
        while not rospy.is_shutdown():
            # determine if we need to publish
            
            cur_state = self.state_buffer.readFromRT()
            prev_plan = self.plan_buffer.readFromRT()
            if cur_state is None:
                continue
            since_last_pub = self.replan_dt if prev_plan is None else (
                cur_state.t - prev_plan.t0).to_sec()
            if since_last_pub >= self.replan_dt:
                if prev_plan is None:
                    u_init = None
                else:
                    u_init = np.zeros((2, self.N))
                    u_init[:, :-1] = prev_plan.nominal_u[:, 1:]

                if self.lead_car_traj_raw is None:
                    continue
                rospy.loginfo(str(self.lead_car_traj_raw))
                sol_x, sol_u, _, _, _, sol_K, _, _ = self.ocp_solver.solve(
                    cur_state.state, controls=u_init, ref_path=self.lead_car_traj_raw, record=True, obs_list=[])
                # print(np.round(sol_x,2))
                # print(np.round(sol_u[1,:],2))
                cur_plan = Plan(sol_x, sol_u, sol_K, cur_state.t, self.replan_dt, self.N)
                self.plan_buffer.writeFromNonRT(cur_plan)
                

    def run(self):
        rospy.spin() 


