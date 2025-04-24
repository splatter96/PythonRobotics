"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""

import numpy as np
import math
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy

from scipy.spatial.transform import Rotation as R

import sys
import pathlib
import time

from gui import Renderer

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from PathPlanning.ReedsSheppPath import reeds_shepp_path_planning as rs

# Parameters
k = 0.0  # look forward gain
# Lfc = 1.7  # [m] look-ahead distance
Lfc = 0.14  # [m] look-ahead distance
# Lfc = 0.54  # [m] look-ahead distance
Kp = 2.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 0.10  # [m] wheel base of vehicle

MAX_STEER = 0.6  # [rad] maximum steering angle
MOTION_RESOLUTION = 0.01  # [m] path interpolate resolution
show_animation = False


# copied from https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.delta = 0
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.front_x = self.x + ((WB / 2) * math.cos(self.yaw))
        self.front_y = self.y + ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        # delta = math.atan(0.5 * math.tan(delta))

        # self.x += self.v * math.cos(self.yaw + sign * delta) * dt
        # self.y += self.v * math.sin(self.yaw + sign * delta) * dt
        self.v += a * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.front_x = self.x + ((WB / 2) * math.cos(self.yaw))
        self.front_y = self.y + ((WB / 2) * math.sin(self.yaw))
        self.delta = delta

    def update_external(self, x, y, yaw, v):
        self.v = v
        self.yaw = yaw
        self.x = x
        self.y = y
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.front_x = self.x + ((WB / 2) * math.cos(self.yaw))
        self.front_y = self.y + ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y, v):
        if v > 0:
            dx = self.x - point_x
            dy = self.y - point_y
        else:
            dx = self.x - point_x
            dy = self.y - point_y
        return math.hypot(dx, dy)

    def __str__(self):
        return f"{self.rear_x} {self.rear_y} {self.front_x} {self.front_y} {self.v}"


class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.front_x = []
        self.front_y = []
        self.yaw = []
        self.v = []
        self.delta = []

    def append(self, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.front_x.append(state.front_x)
        self.front_y.append(state.front_y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.delta.append(state.delta)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:
    def __init__(self, cx, cy, dirs, yaws):
        self.cx = cx
        self.cy = cy
        self.directions = dirs
        self.yaws = yaws
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.front_x - icx for icx in self.cx]
            dy = [state.front_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(
                self.cx[ind], self.cy[ind], state.v
            )
            while True:
                distance_next_index = state.calc_distance(
                    self.cx[ind + 1], self.cy[ind + 1], state.v
                )
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index

            self.old_nearest_point_index = ind

        Lf = k * abs(state.v) + Lfc  # update look ahead distance

        # if state.v < 0:
        #     dx = [state.x - icx for icx in self.cx]
        #     dy = [state.y - icy for icy in self.cy]
        #     # Lf += WB
        # else:
        #     dx = [state.x - icx for icx in self.cx]
        #     dy = [state.y - icy for icy in self.cy]
        #     # Lf += WB
        #
        # d = np.hypot(dx, dy)
        # ind = np.argmin(d)
        #
        ind += 2

        # search look ahead target point index
        current_sign = self.directions[ind]
        while state.calc_distance(self.cx[ind], self.cy[ind], state.v) < Lf and (
            current_sign == self.directions[ind]
        ):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        if current_sign != self.directions[ind]:
            ind -= 1

        self.last_index = ind

        return ind, Lf


def normalise_angle(angle):
    angle = math.atan2(math.sin(angle), math.cos(angle))
    return angle


def stanley_control(state, trajectory, pind, target_speed):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
        tyaw = trajectory.yaws[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        tyaw = trajectory.yaws[-1]
        ind = len(trajectory.cx) - 1

    # check if point in front or behind vehicle
    dx = state.x - tx
    dy = state.y - ty
    in_front = (dy * np.sin(state.yaw) + dx * np.cos(state.yaw)) < 0

    if state.v < 0:
        dx = state.rear_x - tx
        dy = state.rear_y - ty
    else:
        dx = tx - state.front_x
        dy = ty - state.front_y

    crosstrack_error = -(dx * np.sin(state.yaw) - dy * np.cos(state.yaw))

    heading_error = tyaw - state.yaw

    # if state.v < 0:
    #     heading_error = -heading_error

    K = 19.0
    KSOFT = 0.97

    crosstrack_term = np.arctan2((K * crosstrack_error), (KSOFT + abs(target_speed)))

    # print(state.v)

    if state.v < 0:
        crosstrack_term = -crosstrack_term

    sigma_t = crosstrack_term  # + heading_error

    return sigma_t, ind, in_front


def pure_pursuit_steer_control(
    state,
    trajectory,
    pind,
):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    # alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    alpha = math.atan2(ty - state.front_y, tx - state.front_x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind


class PathTracker(Node):
    def __init__(self):
        super().__init__("path_tracker")
        self.first = True
        self.start_pose = None

        # self.target_speed = 0.4 / 3.6  # [m/s]
        self.target_speed = 0.3
        self.target_course = None

        self.renderer = Renderer(None, None, use_wand=True)

        self.pub = self.create_publisher(AckermannDriveStamped, "/car8/cmd_vel", 10)
        self.create_subscription(Pose2D, "/car8/ground_pose", self.odom_callback, 10)
        self.create_subscription(PoseStamped, "car9/pose", self.wand_callback, 10)
        self.create_subscription(Joy, "joy", self.joy_callback, 10)

        self.create_timer(0.1, self.follow_path)

        self.get_logger().info("Initialized PathTracker")

    def joy_callback(self, msg):
        if msg.buttons[5] == 1:
            self.renderer.clicked = True
        else:
            self.renderer.clicked = False

    def wand_callback(self, msg):
        point = msg.pose.position
        orientation = msg.pose.orientation

        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        x_unit_vector = np.array([1, 0, 0])
        direction_vector = r.apply(x_unit_vector)

        # define the ground plane
        planeNormal = np.array([0, 0, 1])
        planePoint = np.array([0, 0, 0])

        ray_point = np.array([point.x, point.y, point.z])
        try:
            hit_point = LinePlaneCollision(
                planeNormal, planePoint, direction_vector, ray_point
            )
        except RuntimeError:
            return

        self.renderer.hit_point = hit_point

    def init_path_planning(self):
        start_x = self.start_pose.x
        start_y = self.start_pose.y
        start_yaw = self.start_pose.theta

        self.state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)

    def plan_path(self):
        goal_x = self.renderer.goal[0]
        goal_y = self.renderer.goal[1]
        goal_yaw = self.renderer.goal_angle

        start_x = self.start_pose.x
        start_y = self.start_pose.y
        start_yaw = self.start_pose.theta

        max_curvature = math.tan(MAX_STEER) / WB
        paths = rs.calc_paths(
            start_x,
            start_y + 0.0,
            start_yaw,
            goal_x,
            goal_y + 0.0,
            goal_yaw,
            max_curvature,
            step_size=MOTION_RESOLUTION,
        )

        # search minimum cost path
        best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
        b_path = paths[best_path_index]

        self.cx = b_path.x
        self.cy = b_path.y
        dirs = b_path.directions
        yaws = b_path.yaw

        # initial state
        state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)
        # self.renderer = Renderer(b_path, self.stop)
        self.renderer.path = b_path
        self.renderer.pause_callback = self.stop

        self.states = States()
        self.states.append(state)
        self.target_course = TargetCourse(self.cx, self.cy, dirs, yaws)
        self.target_ind, _ = self.target_course.search_target_index(state)

    def stop(self):
        drive_msg = AckermannDriveStamped()
        self.pub.publish(drive_msg)
        print("spinning")
        # rclpy.spin_once(self)
        print("spung")

    def follow_path(self):
        if self.renderer.goal is None:
            self.renderer.handle_events()
            self.renderer.render(None, None)
            return

        if self.target_course is None:
            return

        di, self.target_ind, infront = stanley_control(
            self.state, self.target_course, self.target_ind, self.target_speed
        )

        if not infront:
            current_target_speed = -self.target_speed
        else:
            current_target_speed = self.target_speed

        # ai = proportional_control(current_target_speed, self.state.v)

        # self.state.update(ai, di)  # Control vehicle

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = np.rad2deg(di)
        # drive_msg.drive.acceleration = ai
        drive_msg.drive.speed = current_target_speed
        self.pub.publish(drive_msg)

        self.state.delta = di
        self.states.append(self.state)

        self.renderer.render(
            self.states, (self.cx[self.target_ind], self.cy[self.target_ind])
        )
        self.renderer.handle_events()

    def odom_callback(self, msg):
        if self.first:
            self.start_pose = msg
            self.init_path_planning()

        if self.first and self.renderer.goal is not None:
            self.start_pose = msg
            self.plan_path()
            self.first = False
            return

        self.state.update_external(msg.x, msg.y, msg.theta, self.target_speed)
        # self.state.x = msg.x
        # self.state.y = msg.y
        # self.state.yaw = msg.theta
        # self.states.append(self.state)


def main(args=None):
    rclpy.init(args=args)

    tracker = PathTracker()
    rclpy.spin(tracker)

    start_x = 0.0
    start_y = 1.0
    start_yaw = np.deg2rad(90.0)

    goal_x = 1.0
    goal_y = 0.0
    goal_yaw = np.deg2rad(10.0)

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(
        start_x,
        start_y,
        start_yaw,
        goal_x,
        goal_y,
        goal_yaw,
        max_curvature,
        step_size=MOTION_RESOLUTION,
    )

    # search minimum cost path
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    cx = b_path.x
    cy = b_path.y
    dirs = b_path.directions
    yaws = b_path.yaw

    target_speed = 0.1 / 3.6  # [m/s]

    T = 200.0  # max simulation time

    # initial state
    state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)

    lastIndex = len(cx) - 1
    t = 0.0
    states = States()
    states.append(t, state)
    target_course = TargetCourse(cx, cy, dirs, yaws)
    target_ind, _ = target_course.search_target_index(state)

    renderer = Renderer(b_path)

    while (
        T >= t
        and state.calc_distance(
            target_course.cx[lastIndex], target_course.cy[lastIndex], state.v
        )
        > 0.02
    ):
        # check if current waypoint is in front or behind car
        # sign = target_course.directions[target_ind]
        # current_target_speed = sign * target_speed

        # Calc control input
        # di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)
        di, target_ind, infront = stanley_control(
            state, target_course, target_ind, target_speed
        )

        if not infront:
            current_target_speed = -target_speed
        else:
            current_target_speed = target_speed

        ai = proportional_control(current_target_speed, state.v)

        state.update(ai, di)  # Control vehicle

        t += dt
        states.append(t, state)

        renderer.render(states, (cx[target_ind], cy[target_ind]))
        renderer.handle_events()
        time.sleep(0.01)


if __name__ == "__main__":
    print("Pure pursuit path tracking simulation start")
    main()
