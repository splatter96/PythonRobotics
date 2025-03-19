"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""

import numpy as np
import math
import matplotlib.pyplot as plt

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
print(sys.path)

from utils.angle import rot_mat_2d

from PathPlanning.ReedsSheppPath import reeds_shepp_path_planning as rs

# Parameters
k = 0.0  # look forward gain
Lfc = 1.5  # [m] look-ahead distance
Kp = 2.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle

# WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
# VRX = [LF, LF, -LB, -LB, LF]
VRX = [0, 0, -WB, -WB, 0]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

MAX_STEER = 0.6  # [rad] maximum steering angle
MOTION_RESOLUTION = 0.5  # [m] path interpolate resolution
show_animation = True


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        # dx = self.rear_x - point_x
        # dy = self.rear_y - point_y
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)


class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:
    def __init__(self, cx, cy, dirs):
        self.cx = cx
        self.cy = cy
        self.directions = dirs
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(
                    self.cx[ind + 1], self.cy[ind + 1]
                )
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * abs(state.v) + Lfc  # update look ahead distance

        # search look ahead target point index
        current_sign = self.directions[ind]
        while (
            Lf > state.calc_distance(self.cx[ind], self.cy[ind])
            and current_sign == self.directions[ind]
        ):
            print(ind)
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        if current_sign != self.directions[ind]:
            ind -= 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
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
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x,
            y,
            length * math.cos(yaw),
            length * math.sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
        )
        plt.plot(x, y)


def plot_car(x, y, yaw):
    car_color = "-k"
    c, s = math.cos(yaw), math.sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def main():
    #  target course
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    start_x = 47.0
    start_y = 32.0
    start_yaw = np.deg2rad(90.0)

    goal_x = 50.0
    goal_y = 38.0
    goal_yaw = np.deg2rad(-90.0)

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

    # cx = np.load("path_x.npy")[-50:]
    # cy = np.load("path_y.npy")[-50:]
    # dirs = np.load("path_dirs.npy")[-50:]

    target_speed = 1.0 / 3.6  # [m/s]

    T = 200.0  # max simulation time

    # initial state
    state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy, dirs)
    target_ind, _ = target_course.search_target_index(state)

    # while T >= time and lastIndex > target_ind:
    while (
        T >= time
        and state.calc_distance(
            target_course.cx[lastIndex], target_course.cy[lastIndex]
        )
        > 0.2
    ):
        # Calc control input
        di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)

        sign = target_course.directions[target_ind]
        current_target_speed = sign * target_speed
        # if not sign:
        #     current_target_speed = -target_speed
        # else:
        #     current_target_speed = target_speed

        # print(target_ind)
        ai = proportional_control(current_target_speed, state.v)

        state.update(ai, di)  # Control vehicle

        time += dt
        states.append(time, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plot_car(state.x, state.y, state.yaw)
            # plot_arrow(state.x, state.y, state.yaw)
            # plot_arrow(state.rear_x, state.rear_y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.0001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    print("Pure pursuit path tracking simulation start")
    main()
