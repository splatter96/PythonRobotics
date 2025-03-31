from typing import Tuple, Union

import pygame
import numpy as np

PositionType = Union[Tuple[float, float], np.ndarray]

L = 0.17  # [m] wheel base of vehicle
W = 0.08  # width of car


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
    def __init__(self, path):
        self.path = path

        pygame.init()
        pygame.display.set_caption("PurePursuit")  # Also title for i3 config
        panel_size = (1200, 1000)
        # panel_size = (1920, 1920)

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = 385
        # self.sim_surface.scaling = 15
        # self.sim_surface.centering_position = [0.8, -0.8]
        # self.sim_surface.centering_position = [0.2, -0.8]
        self.sim_surface.centering_position = [1.1, -0.7]

        """the world position of the center of the displayed window."""
        self.window_position = np.array([2, 0])

    def render(self, states, lookahead_point):
        self.sim_surface.move_display_window_to(self.window_position)

        self.sim_surface.fill(self.sim_surface.GREY)

        # draw origin of coorindate system
        pygame.draw.line(
            self.sim_surface,
            self.sim_surface.RED,
            self.sim_surface.vec2pix([0, 0]),
            self.sim_surface.vec2pix([0.1, 0]),
            5,
        )
        pygame.draw.line(
            self.sim_surface,
            self.sim_surface.RED,
            self.sim_surface.vec2pix([0, 0]),
            self.sim_surface.vec2pix([0, -0.2]),
            5,
        )
        if len(states.x) < 2:
            return
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
                # for point in zip(states.front_x, states.front_y)
            ],
            5,
        )

        # Render vehicle
        # tire_length, tire_width = 1.0, 0.3
        tire_length, tire_width = 0.035, 0.01

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
            # elif event.type == pygame.MOUSEBUTTONUP:
            #     pix_pos = pygame.Vector2(event.pos)
            #     self.road.vehicles[0].position = np.array(
            #         [*self.sim_surface.pix2pos(pix_pos[0], pix_pos[1])]
            #     )

            self.sim_surface.handle_event(event)


import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from utils.angle import rot_mat_2d
import math

WB = 0.17  # [m] wheel base of vehicle
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
VRX = [0, 0, -WB, -WB, 0]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


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
