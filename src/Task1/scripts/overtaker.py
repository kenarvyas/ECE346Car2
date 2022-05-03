import rospy
import numpy as np

class Overtaker():
    CAR_WIDTH = 0.257 # m
    CAR_LENGTH = 0.43 # m

    # lane x positions (center) meters
    OUTER_LEFT = 3.05
    OUTER_RIGHT = 3.60
    INNER_LEFT = 2.47
    INNER_RIGHT = 2.77

    # merge lane y positions meters
    BEGIN_LANE = 0.51
    END_LANE = 4.39

    def __init__(self) -> None:
        pass

    def pass_car(self, cur_state, lead_car_state):
        lead_car_lane = self._get_lanes(lead_car_state)
        cur_lane = self._get_lanes(cur_state)

        # are we in a merge lane?
        if lead_car_lane == 'X' or cur_lane == 'X':
            rospy.logwarn(f"Not in a merge area. Found in: ({cur_state[0]}, {cur_state[1]})")
            return None
        
        # are we ahead of the lead car?
        if cur_state[1] > (lead_car_state[1] + self.CAR_LENGTH):
            rospy.loginfo("Merging back into lane.")
            return self._merge_back(cur_state, lead_car_state)

        # how should we pass?
        if lead_car_lane == cur_lane:
            rospy.loginfo("Preparing to change lanes.")
            if lead_car_lane is "A":
                rospy.loginfo("Changing lanes to the left.")
                return self._pass_left(cur_state, lead_car_state)
            if lead_car_lane is "C":
                rospy.loginfo("Changing lanes to the left.")
                return self._pass_left(cur_state, lead_car_state)
            if lead_car_lane is "B":
                rospy.loginfo("Changing lanes to the right.")
                return self._pass_right(cur_state, lead_car_state)
            if lead_car_lane is "D":
                rospy.loginfo("Changing lanes to the right.")
                return self._pass_right(cur_state, lead_car_state)
        else:
            rospy.loginfo("Increasing speed.")
            return self._pass_forward(cur_state)

    def _get_lanes(self, state):
        """
        Lane encoding is:
            Outer Right lane = 'A'
            Outer Left lane = 'B'
            inner right lane = 'C'
            inner Left lane = 'D'
            Default lane = 'X'

        Note: All cars move counter-clockwise
        """
        SAFETY_MARGIN = 0.5

        x = state[0]
        y = state[1]

        # are we in the merge zone?
        if  y > self.BEGIN_LANE and y < self.END_LANE:
            # which merge lane are we in?
            if self.OUTER_RIGHT + (SAFETY_MARGIN) * self.CAR_WIDTH > x and x > self.OUTER_LEFT + self.CAR_WIDTH:
                return "A"
            elif self.OUTER_LEFT + self.CAR_WIDTH > x and  x > self.INNER_RIGHT +  self.CAR_WIDTH:
                return "B"
            elif self.INNER_RIGHT + self.CAR_WIDTH > x and x > self.INNER_LEFT +  self.CAR_WIDTH:
                return "C"
            elif self.INNER_LEFT + self.CAR_WIDTH > x and x < self.INNER_LEFT - (SAFETY_MARGIN) * self.CAR_WIDTH:
                return "D"
        return "X"
    
    def _pass_forward(self, state):
        WEIGHT = 2.5

        v = state[2]
        new_vel = self._speed_up(v)
        new_y = self.END_LANE

        new_state = np.array([state[0], new_y, new_vel, state[3]])

        return new_state

    def _pass_left(self, state, lead_state):
        return self._change_lanes(state, lead_state, direction="left")

    def _pass_right(self, state, lead_state):
        return self._change_lanes(state, lead_state, direction="right")

    def _speed_up(self, velocity):
        return velocity
    
    def _change_lanes(self, state, lead_state, direction: str):
        SAFETY_MARGIN = 1.5

        if direction is "left":
            new_x = state[0] - (SAFETY_MARGIN) * self.CAR_WIDTH

        elif direction is "right":
            new_x = state[0] + (SAFETY_MARGIN) * self.CAR_WIDTH

        else:
            rospy.logwarn("Lane change was attempted without specifying a lane. No change was executed.")
            return state
        
        new_y = lead_state[1]
        new_state = np.array([new_x, new_y, state[2], state[3]])
        return new_state

    def _merge_back(self, state, lead_state):
        lead_car_lane = self._get_lanes(lead_state)
        cur_lane = self._get_lanes(state)

        # are we in a merge lane?
        if lead_car_lane == 'X' or cur_lane == 'X':
            return state

        # How should we merge?
        if lead_car_lane != cur_lane:
            if lead_car_lane is "A" or lead_car_lane is "C":
                return self._pass_right(state, lead_state)
            if lead_car_lane is "B" or lead_car_lane is "D":
                return self._pass_left(state, lead_state)
        
        return state
