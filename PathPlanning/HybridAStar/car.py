"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import sys
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from math import cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    x = np.array(x_list)
    y = np.array(y_list)

    query_points = np.vstack([x, y]).T

    # find all obstacle ids potentially colliding with the path
    ids = kd_tree.query_ball_point(query_points, BUBBLE_R)

    for i, (i_x, i_y, i_yaw) in enumerate(zip(x_list, y_list, yaw_list)):
        if not ids[i]:
            continue

        # do a precise rectanglular check for collisions
        if rectangle_check(
            i_x, i_y, i_yaw, [ox[j] for j in ids[i]], [oy[j] for j in ids[i]]
        ):
            return True  # collision

    return False  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return True  # collision

    return False  # no collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(
            x,
            y,
            length * cos(yaw),
            length * sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
            alpha=0.4,
        )


def plot_car(x, y, yaw):
    car_color = "-k"
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


# convert angle to be between -pi and pi
def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw = pi_2_pi(yaw + distance * tan(steer) / L)  # distance/2

    return x, y, yaw


def main():
    x, y, yaw = 0.0, 0.0, 1.0
    plt.axis("equal")
    plot_car(x, y, yaw)
    plt.show()


if __name__ == "__main__":
    main()
