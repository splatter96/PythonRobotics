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
import time

from typing import List, Tuple, Union, TYPE_CHECKING

PositionType = Union[Tuple[float, float], np.ndarray]

import pygame

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
print(sys.path)

from utils.angle import rot_mat_2d

from PathPlanning.ReedsSheppPath import reeds_shepp_path_planning as rs

# Parameters
k = 0.0  # look forward gain
Lfc = 1.7  # [m] look-ahead distance
Kp = 2.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle

L = WB  # length of car
# WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
# VRX = [LF, LF, -LB, -LB, LF]
VRX = [0, 0, -WB, -WB, 0]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

MAX_STEER = 0.6  # [rad] maximum steering angle
# MAX_STEER = 0.3  # [rad] maximum steering angle
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
show_animation = False


class WorldSurface(pygame.Surface):
    """A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area."""

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(
        self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType
    ) -> None:
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        # return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])
        return self.pix(x - self.origin[0]), self.pix(self.origin[1] - y)

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return (
            -margin < x < self.get_width() + margin
            and -margin < y < self.get_height() + margin
        )

    def move_display_window_to(self, position: PositionType) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling,
            ]
        )

    def handle_event(self, event: pygame.event.EventType) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PAGEUP:
                self.scaling *= 1 / self.SCALING_FACTOR
                print(f"Scaling {self.scaling}")
            if event.key == pygame.K_PAGEDOWN:
                self.scaling *= self.SCALING_FACTOR
                print(f"Scaling {self.scaling}")
            if event.key == pygame.K_LEFT:
                self.centering_position[0] -= self.MOVING_FACTOR
                print(f"Center x {self.centering_position[0]}")
            if event.key == pygame.K_RIGHT:
                self.centering_position[0] += self.MOVING_FACTOR
                print(f"Center x {self.centering_position[0]}")
            if event.key == pygame.K_UP:
                self.centering_position[1] -= self.MOVING_FACTOR
                print(f"Center y {self.centering_position[1]}")
            if event.key == pygame.K_DOWN:
                self.centering_position[1] += self.MOVING_FACTOR
                print(f"Center y {self.centering_position[1]}")


class Renderer:
    def __init__(self, vehicle, path):
        self.vehicle = vehicle
        self.path = path

        pygame.init()
        pygame.display.set_caption("PurePursuit")  # Also title for i3 config
        panel_size = (1200, 1000)
        # panel_size = (1920, 1920)

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        # self.sim_surface.scaling = 385
        self.sim_surface.scaling = 15
        # self.sim_surface.centering_position = [0.8, -0.8]
        self.sim_surface.centering_position = [0.2, -0.8]

        """the world position of the center of the displayed window."""
        self.window_position = np.array([2, 0])

    def render(self, states, lookahead_point):
        self.sim_surface.move_display_window_to(self.window_position)

        self.sim_surface.fill(self.sim_surface.GREY)

        # Render planned path
        pygame.draw.lines(
            self.sim_surface,
            self.sim_surface.RED,
            False,
            [
                self.sim_surface.vec2pix(point)
                # for point in lane.lanelet.polygon.vertices
                for point in zip(self.path.x, self.path.y)
            ],
            5,
        )

        # Render followed path
        pygame.draw.lines(
            self.sim_surface,
            self.sim_surface.BLUE,
            False,
            [
                self.sim_surface.vec2pix(point)
                # for point in lane.lanelet.polygon.vertices
                for point in zip(states.x, states.y)
            ],
            5,
        )

        # Render vehicle
        tire_length, tire_width = 1.0, 0.3

        # Vehicle rectangle
        length = L + 2 * tire_length
        vehicle_surface = pygame.Surface(
            (self.sim_surface.pix(length), self.sim_surface.pix(length)),
            flags=pygame.SRCALPHA,
        )  # per-pixel alpha
        rect = (
            self.sim_surface.pix(tire_length),
            self.sim_surface.pix(length / 2 - W / 2),
            self.sim_surface.pix(L),
            self.sim_surface.pix(W),
        )
        # pygame.draw.rect(vehicle_surface, self.sim_surface.WHITE, rect, 0)
        pygame.draw.rect(vehicle_surface, self.sim_surface.BLACK, rect, 1)

        tire_positions = [
            [
                self.sim_surface.pix(tire_length),
                self.sim_surface.pix(length / 2 - W / 2),
            ],
            [
                self.sim_surface.pix(tire_length),
                self.sim_surface.pix(length / 2 + W / 2),
            ],
            [
                self.sim_surface.pix(length - tire_length),
                self.sim_surface.pix(length / 2 - W / 2),
            ],
            [
                self.sim_surface.pix(length - tire_length),
                self.sim_surface.pix(length / 2 + W / 2),
            ],
        ]
        # tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
        tire_angles = [0, 0, states.delta[-1], states.delta[-1]]
        for tire_position, tire_angle in zip(tire_positions, tire_angles):
            tire_surface = pygame.Surface(
                (self.sim_surface.pix(tire_length), self.sim_surface.pix(tire_length)),
                pygame.SRCALPHA,
            )
            rect = (
                0,
                self.sim_surface.pix(tire_length / 2 - tire_width / 2),
                self.sim_surface.pix(tire_length),
                self.sim_surface.pix(tire_width),
            )
            pygame.draw.rect(tire_surface, self.sim_surface.BLACK, rect, 0)
            self.blit_rotate(
                vehicle_surface,
                tire_surface,
                tire_position,
                np.rad2deg(tire_angle),
            )

        h = states.yaw[-1]

        # front_x = states.x[-1] - ((L / 2) * math.cos(states.yaw[-1]))
        # front_y = states.y[-1] - ((L / 2) * math.sin(states.yaw[-1]))
        # position = [*self.sim_surface.pos2pix(front_x, front_y)]

        position = [*self.sim_surface.pos2pix(states.x[-1], states.y[-1])]

        self.blit_rotate(self.sim_surface, vehicle_surface, position, np.rad2deg(h))

        # Render the lookahead point
        pygame.draw.circle(
            self.sim_surface,
            self.sim_surface.GREEN,
            self.sim_surface.pos2pix(*lookahead_point),
            4,
        )

        # Render the point calculate for the front axle
        pygame.draw.circle(
            self.sim_surface,
            self.sim_surface.RED,
            self.sim_surface.pos2pix(states.front_x[-1], states.front_y[-1]),
            4,
        )

        self.screen.blit(self.sim_surface, (0, 0))
        pygame.display.flip()

    def blit_rotate(
        self,
        surf,
        image,
        pos,
        angle,
        origin_pos=None,
        show_rect=False,
    ) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (
            min(box_rotate, key=lambda p: p[0])[0],
            min(box_rotate, key=lambda p: p[1])[1],
        )
        max_box = (
            max(box_rotate, key=lambda p: p[0])[0],
            max(box_rotate, key=lambda p: p[1])[1],
        )

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0],
            pos[1] - origin_pos[1] - max_box[1] + pivot_move[1],
        )
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = True
                while paused:
                    pygame.time.wait(1000)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            paused = False
            elif event.type == pygame.MOUSEBUTTONUP:
                pix_pos = pygame.Vector2(event.pos)
                self.road.vehicles[0].position = np.array(
                    [*self.sim_surface.pix2pos(pix_pos[0], pix_pos[1])]
                )

            self.sim_surface.handle_event(event)


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

    def calc_distance(self, point_x, point_y, v):
        if v < 0:
            dx = self.front_x - point_x
            dy = self.front_y - point_y
        else:
            dx = self.rear_x - point_x
            dy = self.rear_y - point_y
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
        self.t = []
        self.delta = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.front_x.append(state.front_x)
        self.front_y.append(state.front_y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.delta.append(state.delta)
        self.t.append(t)


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
        # print(f"{state.v=}")
        # if self.old_nearest_point_index is None:
        #     # search nearest point index
        #     dx = [state.front_x - icx for icx in self.cx]
        #     dy = [state.front_y - icy for icy in self.cy]
        #     d = np.hypot(dx, dy)
        #     ind = np.argmin(d)
        #     self.old_nearest_point_index = ind
        # else:
        #     ind = self.old_nearest_point_index
        #     distance_this_index = state.calc_distance(
        #         self.cx[ind], self.cy[ind], state.v
        #     )
        #     while True:
        #         distance_next_index = state.calc_distance(
        #             self.cx[ind + 1], self.cy[ind + 1], state.v
        #         )
        #         if distance_this_index < distance_next_index:
        #             break
        #         ind = ind + 1 if (ind + 1) < len(self.cx) else ind
        #         distance_this_index = distance_next_index
        #
        #     print(f"{distance_this_index=}")
        #     print(f"{distance_next_index=}")
        #     self.old_nearest_point_index = ind
        #
        Lf = k * abs(state.v) + Lfc  # update look ahead distance

        if state.v > 0:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            Lf += WB
        else:
            dx = [state.front_x - icx for icx in self.cx]
            dy = [state.front_y - icy for icy in self.cy]
            Lf += WB
        d = np.hypot(dx, dy)
        ind = np.argmin(d)

        # search look ahead target point index
        current_sign = self.directions[ind]
        while (
            state.calc_distance(self.cx[ind], self.cy[ind], state.v) < Lf
            and current_sign == self.directions[ind]
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

    if state.v < 0:
        dx = state.rear_x - tx
        dy = state.rear_y - ty
    else:
        dx = tx - state.front_x
        dy = ty - state.front_y

    front_axle_vec = [-np.cos(state.yaw), -np.sin(state.yaw)]
    crosstrack_error = np.dot([dx, dy], front_axle_vec)

    crosstrack_error = -(dx * np.sin(state.yaw) - dy * np.cos(state.yaw))

    # heading_error = normalise_angle(tyaw - state.yaw - np.pi * 0.5)
    # heading_error = normalise_angle(tyaw - state.yaw)
    if state.v < 0:
        heading_error = tyaw - state.yaw
    else:
        heading_error = tyaw - state.yaw

    K = 1.0
    KSOFT = 1.0

    crosstrack_term = np.arctan2((K * crosstrack_error), (KSOFT + abs(target_speed)))
    # heading_term = normalise_angle(heading_error)
    heading_term = heading_error

    sigma_t = crosstrack_term  # + heading_term
    if state.v < 0:
        sigma_t = -sigma_t

    return sigma_t, ind


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
    start_y = 22.0
    # start_x = 0.0
    # start_y = 0.0
    start_yaw = np.deg2rad(-40.0)

    goal_x = 50.0
    goal_y = 38.0
    # goal_x = 50.0
    # goal_y = -10.0
    goal_yaw = np.deg2rad(40.0)

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

    # cx = np.load("path_x.npy")[-50:]
    # cy = np.load("path_y.npy")[-50:]
    # dirs = np.load("path_dirs.npy")[-50:]

    target_speed = 1.0 / 3.6  # [m/s]

    T = 200.0  # max simulation time

    # initial state
    state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)

    lastIndex = len(cx) - 1
    t = 0.0
    states = States()
    states.append(t, state)
    target_course = TargetCourse(cx, cy, dirs, yaws)
    target_ind, _ = target_course.search_target_index(state)

    renderer = Renderer(None, b_path)

    # while T >= t and lastIndex > target_ind:
    while (
        T >= t
        and state.calc_distance(
            target_course.cx[lastIndex], target_course.cy[lastIndex], state.v
        )
        > 0.2
    ):
        sign = target_course.directions[target_ind]
        # print(f"{sign=}")
        current_target_speed = sign * target_speed

        # Calc control input
        # di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)
        di, target_ind = stanley_control(
            state, target_course, target_ind, current_target_speed
        )

        # if not sign:
        #     current_target_speed = -target_speed
        # else:
        #     current_target_speed = target_speed

        # print(target_ind)
        ai = proportional_control(current_target_speed, state.v)

        state.update(ai, di)  # Control vehicle

        t += dt
        states.append(t, state)

        renderer.render(states, (cx[target_ind], cy[target_ind]))
        renderer.handle_events()
        time.sleep(0.01)

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
