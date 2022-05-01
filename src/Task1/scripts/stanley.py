import numpy as np

class Stanley():
    def __init__(self, gain, ref_path=None, wheel_base=0.257, kp = 1, ki = 0.1) -> None:
        self.ref_path = ref_path
        self.k = gain
        self.L = wheel_base

        self.kp = kp
        self.ki = ki

        self.accum_error = 0

    def solve(self, cur_state):
        direction = self.stanley_control(cur_state)
        speed = self.speed_control(cur_state)
        return speed, direction

    def stanley_control(self, cur_pos):
        error, index = self.get_steering_error(cur_pos, self.ref_path)
        
        cur_yaw = cur_pos[3]
        heading_yaw = np.arctan2(self.ref_path[1,-1] - self.ref_path[1,0], self.ref_path[0,-1] - self.ref_path[0,0])
        
        heading_error = self.normalize_angle(heading_yaw - cur_yaw)
        tracking_error = np.arctan2(self.k * error, cur_pos[2])

        steering_angle = heading_error + tracking_error

        # steering_angle = np.tan(steering_angle)

        return np.tan(steering_angle)
    
    def get_steering_error(self, current, trajectory):
        x = current[0]
        y = current[1]
        yaw = current[3]

        trajx = trajectory[0,:]
        trajy = trajectory[1,:]

        fx = x + self.L * np.cos(yaw)
        fy = y + self.L * np.sin(yaw)
        
        dx = [fx - x for x in trajx]
        dy = [fy - y for y in trajy]
        d = np.hypot(dx, dy)

        loc  = np.argmin(d)

        front_axle = [-np.cos(yaw + np.pi/2),
                      -np.sin(yaw + np.pi/2)]
        error = np.dot([dx[loc], dy[loc]], front_axle)

        return error, loc

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        
        return angle
    
    def speed_control(self, current):
        DISCOUNT = 0.9

        error = np.linalg.norm(self.ref_path[:2, -1] - current[:2])
        self.accum_error += DISCOUNT * error

        return self.kp * error + self.ki * self.accum_error
