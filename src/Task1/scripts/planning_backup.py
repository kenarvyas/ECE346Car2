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
from pyspline.pyCurve import Curve
from stanley import Stanley
from rc_control_msgs.msg import RCControl
import yaml, csv
import time
import matplotlib.pyplot as plt

import queue
import rospy
from dynamic_reconfigure.server import Server

from traj_msgs.msg import TrajMsg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from rc_control_msgs.msg import RCControl
from std_msgs.msg import Bool

import numpy as np
from pyspline.pyCurve import Curve
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import yaml
from threading import Lock


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

class Trajectory(object):
    """
    Creata lists of x, y, psi, vel using  fitting 
    based on the x, y, psi, vel of the Trajectory message
    """

    def __init__(self, msg, L=0.257):
        '''
        Decode the ros message and apply cubic interpolation 
        '''
        self.t_0 = msg['t0']
        self.dt = msg['dt']
        self.step = msg['step']
        self.t = np.arange(self.step) * self.dt
        self.t_final = self.step * self.dt

        # Convert the trajectory planned at rear axis to the front axis
        x_f = msg['x'] + np.cos(msg['psi']) * L
        y_f = msg['y'] + np.sin(msg['psi']) * L

        # use b-spline to fit trajectory
        self.ref_traj_f = Curve(x=msg['x'], y=msg['y'], k=3)

        # use cubic spline to fit time dependent v, psi, pos (x, y)
        self.ref_x_f = CubicSpline(self.t, np.array(x_f))
        self.ref_y_f = CubicSpline(self.t, np.array(y_f))
        self.ref_v_r = CubicSpline(self.t, np.array(msg['vel']))
        self.ref_psi_r = CubicSpline(self.t, np.array(msg['psi']))
        self.loop = False

        self.length = self.ref_traj_f.getLength()
        theta_sample = np.linspace(0, 1, self.step * 10,
                                   endpoint=False) * self.length
        self.data, slope = self.interp(theta_sample)

    def get_closest_pt(self, x, y, psi, t):
        '''
        Points have [2xn] shape
        '''

        s, _ = self.ref_traj_f.projectPoint(np.array([x, y]), eps=1e-3)
        closest_pt = self.ref_traj_f.getValue(s)

        # Instead of using the slope of the spline, use the actual psi from the planner
        deri = self.ref_traj_f.getDerivative(s)
        slope_spline = np.arctan2(deri[1], deri[0])

        slope = self.get_psi_ref(t)

        error_spline = np.sin(slope_spline) * (x - closest_pt[0]) \
                        - np.cos(slope_spline) * (y - closest_pt[1])

        error = np.sin(psi) * (x - closest_pt[0]) \
                        - np.cos(psi) * (y - closest_pt[1])
        
        return slope, slope_spline, error, error_spline

    def get_pos_ref(self, t):
        dt = t.to_sec() - self.t_0
        return self.ref_x_f(dt), self.ref_y_f(dt)

    def get_v_ref(self, t):
        dt = t.to_sec() - self.t_0
        return self.ref_v_r(dt)

    def get_psi_ref(self, t):
        dt = t.to_sec() - self.t_0
        return self.ref_psi_r(dt)

    def _interp_s(self, s):
        '''
        Given a list of s (progress since start), return corresponing (x,y) points  
        on the track. In addition, return slope of trangent line on those points
        '''
        n = len(s)

        interp_pt = self.ref_traj_f.getValue(s)
        slope = np.zeros(n)

        for i in range(n):
            deri = self.ref_traj_f.getDerivative(s[i])
            slope[i] = np.arctan2(deri[1], deri[0])
        return interp_pt.T, slope

    def interp(self, theta_list):
        '''
        Given a list of theta (progress since start), return corresponing (x,y) points  
        on the track. In addition, return slope of trangent line on those points
        '''
        if self.loop:
            s = np.remainder(theta_list, self.length) / self.length
        else:
            s = np.array(theta_list) / self.length
            s[s > 1] = 1
        return self._interp_s(s)


        
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

        # real time buffer for planned trajectory
        self.traj_buffer = RealtimeBuffer()

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
        
        self.prev_t = None
        self.d_f = 0.257
        self.L = 0.257
        # distance from pose center to the rear axis (m)
        self.d_r = self.d_f - self.L
    
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
            dt = .01
            v = 0

        cur_X = np.array([x, y, v, psi])
        self.lead_car_state_buffer.writeFromNonRT(State(cur_X, cur_t))
        

        if self.lead_car_center_line is None:
            self.lead_car_center_line = np.array([[cur_X[0], cur_X[1], cur_X[2], cur_X[3]]]).T
        else:
            self.lead_car_center_line = np.append(self.lead_car_center_line,np.array([[cur_X[0], cur_X[1], cur_X[2], cur_X[3]]]).T, axis=1) # append most recent x and y
        if len(self.lead_car_center_line) > 10:
            self.traj_buffer.writeFromNonRT(Trajectory({'t0': cur_t, 'x': self.lead_car_center_line[0,:], 'y': self.lead_car_center_line[1, :], 'vel':self.lead_car_center_line[2,:], 'psi':  self.lead_car_center_line[3, :], 'dt': dt, 'step': len(self.lead_car_center_line[0,:]), }))

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

        # position of front axis
        x_f = x + np.cos(psi) * self.d_f
        y_f = y + np.sin(psi) * self.d_f

        # linear velocity
        if self.prev_t is not None:
            dx = x_f - self.prev_x
            dy = y_f - self.prev_y
            dt = cur_t.to_sec() - self.prev_t

            # assert dt != 0
            v = np.sqrt(dx * dx + dy * dy) / dt
            rospy.loginfo(f"V = {v}")

        else:
            v = None
        if v is not None:
            cur_traj = self.traj_buffer.readFromRT()

            if cur_traj is not None:
                ref_v = cur_traj.get_v_ref(cur_t)
                delta = self.lateral_control(cur_traj, x_f, y_f, psi, v, cur_t, dt)
                d = self.longitudial_control(cur_traj.get_pos_ref(cur_t),
                                             [x_f, y_f], v, ref_v, psi, dt)
                d = d + abs(delta) * 0.01
                # constraint for acceleration
                d = np.clip(
                    (d - self.prev_d) / dt, -0.1, 0.1) * dt + self.prev_d

                self.publish_control(cur_t, d, delta)

                self.prev_d = d

        self.prev_x = x_f
        self.prev_y = y_f
        self.prev_t = cur_t.to_sec()

    def publish_control(self, cur_t, d, delta):
        control = RCControl()
        control.header.stamp = cur_t
        #! MAP VALUE OF STANLEY OUTPUT TO THROTTLE AND STEERING
        control.throttle = np.clip(d, -1.0, 1.0)
        control.steer = np.clip(delta, -1.0, 1.0)
        control.reverse = False
        self.control_pub.publish(control)

    def lateral_control(self, cur_traj, x_f, y_f, psi, v, cur_t, dt):
        """
        slope: reference psi from ilqr
        slope_spline: reference psi from where the car is wrt to reference spline
        error: error of car's current psi and slope
        error_spline: error of car's curretn psi and slope_spline
        """
        slope, slope_spline, error, error_spline = cur_traj.get_closest_pt(x_f, y_f, psi, cur_t)

        # theta_e corrects the heading error
        e_psi = self.normalize_angle(slope_spline - psi)
        theta_e = e_psi* self.p_psi
        
        # theta_d contouring error
        # cap v so that this will not go to inf
        theta_d = np.arctan2(self.p_lat * error_spline, 1+v)
        # print(e_psi, error_spline)
        if self.prev_epsi is not None:
            epsi_dev = (e_psi - self.prev_epsi) / dt
        else:
            epsi_dev = e_psi / dt
        self.prev_epsi = e_psi

        # Steering control
        delta = theta_e + theta_d - epsi_dev * self.d_psi

        # print("theta_e: {:.3f}, theta_d: {:.3f}, delta: {:.3f}".format(theta_e, theta_d, delta))

        return -1.0 * delta

    def longitudial_control(self, pos_ref, cur_pos, cur_v, ref_v, cur_psi, dt):
        e_v = ref_v - cur_v
        # print("e_v:\t\t", e_v)

        # reset the integral term
        if abs(e_v) < 0.002:
            self.e_int = 0
        else:
            self.e_int = np.clip(self.e_int + e_v, -self.i_sat, self.i_sat)

        # derivate term
        if self.prev_ev is not None:
            ev_dev = (e_v - self.prev_ev) / dt
        else:
            ev_dev = e_v / dt
        self.prev_ev = e_v

        # # check to see where ground truth is
        dpos = np.array(cur_pos) - np.array(pos_ref)
        e_pos = np.sqrt(dpos @ dpos.T)

        # # check to see if the car is in front or behind the ground truth
        # heading_angle = np.arctan2(dpos[1], dpos[0])
        # is_front = False
        # if abs(
        #         np.arctan2(np.sin(heading_angle - cur_psi),
        #                    np.cos(heading_angle - cur_psi))) < np.pi * 0.5:
        #     is_front = True

        d = self.p_lon*e_v                  \
            + self.i_lon*self.e_int         \
            + e_pos * self.p_lon_complement \
            - ev_dev * self.d_lon

        # if is_front:
        #     d = 0

        return d
                
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        
        return angle

    def run(self):
        rospy.spin() 