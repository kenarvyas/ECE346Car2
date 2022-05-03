import numpy as np

class Stanley():
    def __init__(self, gain, ref_path=None, wheel_base=0.257, kp = 1, ki = 0.1, k_soft=0, k_steer=0) -> None:
        self.ref_path = ref_path
        self.k = gain
        self.L = wheel_base
        self.last_target_idx = None
        self.last_delta = None

        self.kp = kp
        self.ki = ki
        self.k_soft = k_soft
        self.k_steer = k_steer

        self.accum_error = 0

    def solve(self, cur_state):
        direction, curr_target_idx = self.stanley_control(cur_state, self.ref_path[0], self.ref_path[1], self.ref_path[2], self.last_target_idx)
        self.last_target_idx = curr_target_idx
        speed = self.speed_control(cur_state, curr_target_idx)
        return (speed, direction), curr_target_idx

    def stanley_control(self, state, cx, cy, cyaw, last_target_idx):
        """
        Stanley steering control.
        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index(state, cx, cy)

        if last_target_idx and last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx  

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - state[3])
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, state[2] + self.k_soft)
        # Steering control
        delta = theta_e + theta_d

        if False and self.last_delta:
            delta += self.k_steer*(delta - self.last_delta)

        self.last_delta = delta

        return delta, current_target_idx
    

    def calc_target_index(self, state, cx, cy):
        """
        Compute index in the trajectory list of the target.
        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = state[0] + self.L * np.cos(state[3])
        fy = state[1] + self.L * np.sin(state[3])

        # Search nearest point index
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(state[3] + np.pi / 2),
                        -np.sin(state[3] + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        
        return angle
    
    def speed_control(self, current, target_id):
        DISCOUNT = 0.9

        error = np.linalg.norm(self.ref_path[:2, target_id] - current[:2])
        self.accum_error *= DISCOUNT
        self.accum_error += error

        return self.kp * error + self.ki * self.accum_error

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path[-50:]
        self.last_target_idx = None
        self.accum_error = 0
