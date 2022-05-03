#!/usr/bin/env python
import threading
from collections import deque

import rospy
import numpy as np
from iLQR import iLQR, Track, EllipsoidObj
from realtime_buffer import RealtimeBuffer
from traj_msgs.msg import TrajMsg
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from stanley_backup import Stanley
from rc_control_msgs.msg import RCControl
import yaml, csv
import time
import matplotlib.pyplot as plt

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
        # self.ocp_solver = iLQR(params=self.params)
        gain = .5
        self.ocp_solver = Stanley(gain)

        rospy.loginfo("Successfully initialized the solver with horizon " +
                    str(self.T) + "s, and " + str(self.N) + " steps.")

        self.state_buffer = RealtimeBuffer()
        self.plan_buffer = RealtimeBuffer()
        self.lead_car_state_buffer = RealtimeBuffer()
        self.lead_car_center_line = None
        self.lead_car_traj = None

        self.i = -1

        # set up publiser to the reference trajectory and subscriber to the pose
        self.control_pub = rospy.Publisher(control_topic, RCControl, queue_size=1)

        self.pose_sub = rospy.Subscriber(pose_topic,
                                        Odometry,
                                        self.odom_sub_callback,
                                        queue_size=1)

        self.lead_car_pose_sub = rospy.Subscriber(lead_car_pose_topic,
                                        Odometry,
                                        self.lead_car_odom_sub_callback,
                                        queue_size=1)
    
        # start planning thread
        # threading.Thread(target=self.ilqr_pub_thread).start()

    def lead_car_odom_sub_callback(self, odomMsg):
        """
        Subscriber callback function of the lead robot pose
        """
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

        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1]]])
            # self.lead_car_headings = np.array([cur_X[3]]) 
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line,[[cur_X[0], cur_X[1]], cur_X[3]], axis=0) # append most recent x and y
            # self.lead_car_headings = np.append(self.lead_car_headings, [cur_X[3]], axis=0)

        # if len(self.lead_car_center_line) > 45:
        #     # create nominal trajectory not counting last 10 to keep following vehicle behind (max len of nominal traj is 300)
        #     if len(self.lead_car_center_line) > 345:
        #         nominal_traj = self.lead_car_center_line[-300:-30,:].T
        #     else:
        #         nominal_traj = self.lead_car_center_line[:-30,:].T

        #     # turn nominal trajectory into track so iLQR can use it (given to ilqr in ilqr sub function below)
        #     self.lead_car_traj = Track(center_line=nominal_traj,
        #                     width_left=self.params['track_width_L'],
        #                     width_right=self.params['track_width_R'],
        #                     loop=True)

    def odom_sub_callback(self, odomMsg):
        """
        Subscriber callback function of the robot pose
        """
        

        cur_t = odomMsg.header.stamp

        # self.publish_control(0, [0,0], cur_t)
        # rospy.loginfo(-1 + self.i*.1)
        # self.i += 1
        # return
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
        rospy.loginfo(cur_X)
        # obtain the latest plan
        # last_plan = self.plan_buffer.readFromRT()
        # if last_plan is not None:
        #     # get the control policy
        #     X_k, u_k, K_k = last_plan.get_policy(cur_t)
        #     u = u_k
        #     # u = u_k+ K_k@(cur_X - X_k)           
        #     self.publish_control(v, u, cur_t)

        self.ocp_solver.ref_path = self.lead_car_center_line
        #np.array([self.lead_car_center_line, self.lead_car_headings])
        sol_u = self.ocp_solver.solve(cur_X)

        rospy.loginfo(sol_u[1])

        self.publish_control(v, sol_u, cur_t)

        # plt.figure()
        # plt.plot(self.lead_car_center_line[0,:], self.lead_car_center_line[1,:], color='orange')
        # plt.scatter(cur_X[0], cur_X[1], color="blue")
        # plt.savefig("traj_stan/test_vmax01_"+str(i))
        
        # write the new pose to the buffer
        self.state_buffer.writeFromNonRT(State(cur_X, cur_t))

    def publish_control(self, v, u, cur_t):
        control = RCControl()
        control.header.stamp = cur_t
        a = u[0]
        delta = -u[1]
        
        if a<0:
            d = a/10-0.5
        else:
            temp = np.array([v**3, v**2, v, a**3, a**2, a, v**2*a, v*a**2, v*a, 1])
            d = temp@self.d_open_loop
            d = d+min(delta*delta*0.5,0.05)

        control.throttle = np.clip(d, -1.0, 0.2)
        control.steer = np.clip(delta, -1.0, 1.0)
        control.reverse = False
        self.control_pub.publish(control)

    # def ilqr_pub_thread(self):
    #     time.sleep(5)
    #     rospy.loginfo("iLQR Planning publishing thread started")
    #     i = 0
    #     follow_x = []
    #     follow_y = []
    #     while not rospy.is_shutdown():
    #         # determine if we need to publish
            
    #         cur_state = self.state_buffer.readFromRT()
    #         prev_plan = self.plan_buffer.readFromRT()
    #         if cur_state is None:
    #             continue
    #         since_last_pub = self.replan_dt if prev_plan is None else (
    #             cur_state.t - prev_plan.t0).to_sec()
    #         if since_last_pub >= self.replan_dt:
    #             if prev_plan is None:
    #                 u_init = None
    #             else:
    #                 u_init = np.zeros((2, self.N))
    #                 u_init[:, :-1] = prev_plan.nominal_u[:, 1:]
                
    #             if self.lead_car_traj is None:
    #                 continue

    #             # self.ocp_solver.ref_path = self.lead_car_traj # Set iLQR reference track to be the traj of lead car
    #             self.ocp_solver.ref_path = self.lead_car_center_line

    #             # sol_x, sol_u, _, _, _, sol_K, _, _ = self.ocp_solver.solve(
    #             #     cur_state.state, u_init, record=True, obs_list=[])
    #             sol_u = self.ocp_solver.solve(cur_state.state)

    #             if i%25 == 0:
    #                 pass
    #                 # plt.figure()
    #                 # self.lead_car_traj.plot_track_center()
    #                 # #rospy.loginfo("sol_x shape " + str(sol_x.shape))
    #                 # plt.plot(sol_x[0,:], sol_x[1,:], color="orange")
    #                 # plt.scatter(follow_x, follow_y, color="blue")
    #                 # plt.savefig("traj/test_vmax01_"+str(i))
    #                 # follow_x, follow_y = [], []
    #             i+=1
    #             rospy.loginfo(i)
    #             follow_x.append(cur_state.state[0])
    #             follow_y.append(cur_state.state[1])
    #             #rospy.loginfo("curstate " + str(cur_state.state))
    #             # cur_plan = Plan(sol_x, sol_u, sol_K, cur_state.t, self.replan_dt, self.N)
    #             cur_plan = Plan([], sol_u, [], cur_state.t, self.replan_dt, self.N)
    #             self.plan_buffer.writeFromNonRT(cur_plan)
                

    def run(self):
        rospy.spin() 