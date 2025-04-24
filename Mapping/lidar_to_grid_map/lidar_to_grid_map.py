"""

LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts

"""

import math
from collections import deque

import sys
sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
sys.path.append("/home/paul/Documents/PhD/RL/MARL_CAVs_commonroad/highway-env")

import cv2

import pygame

import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose2D

from rclpy.qos import qos_profile_sensor_data

from highway_env.road.graphics import WorldSurface

EXTEND_AREA = 1.0
class Renderer:
    def __init__(self,):

        pygame.init()
        pygame.display.set_caption("Highway-env")  # Also title for i3 config
        # panel_size = (1000, 600)
        # panel_size = (1200, 800)
        # panel_size = (1920, 1200)
        panel_size = (1920, 1920)

        self.apply_colormap = True
        self.image = None
        self.grid_resolution = 1

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        # self.sim_surface.scaling = 169.0
        # self.sim_surface.scaling = 377.36
        # self.sim_surface.scaling = 320.76
        # self.sim_surface.scaling = 389.402
        self.sim_surface.scaling = 385
        # self.sim_surface.centering_position = [0.7, -0.5]
        self.sim_surface.centering_position = [0.8, -0.8]
        # self.sim_surface.scaling = 4.23
        # self.sim_surface.centering_position = [-0.71, -0.5]

        """the world position of the center of the displayed window."""
        self.window_position = np.array([2, 0])

        self.paused = False
        self.points = None

    def render_lidar(self):
        points = [self.sim_surface.vec2pix(point) for point in self.points]
        pygame.draw.lines(self.sim_surface, (255, 100, 100), False, points, 5)

    def render(self):
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

        if self.points is not None:
            self.render_lidar()

        self.screen.blit(self.sim_surface, (0, 0))

        if self.image is not None:
            if self.apply_colormap:
                image_alpha = cv2.applyColorMap(self.image, cv2.COLORMAP_JET)
                image_alpha = cv2.cvtColor(image_alpha, cv2.COLOR_BGR2BGRA)

                # print(image_alpha)

                # # use original grayscale as alpha channel
                # image_alpha[:, :, 3] = self.image[:, :, 0]
            else:
                image_alpha = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)

            py_img = pygame.image.frombuffer(
                image_alpha.tostring(),
                self.image.shape[1::-1],
                "BGRA" if self.apply_colormap else "RGBA",
            )

            # scale image to corrrect physical world size
            #physical_size = np.array([3, 3])
            physical_size = np.array([py_img.get_width(), py_img.get_height()]) * self.grid_resolution
            # print(physical_size)
            physical_size_pixels = physical_size * self.sim_surface.scaling #* (1.0/self.grid_resolution)
            py_img = pygame.transform.scale(py_img, physical_size_pixels)

            py_img = pygame.transform.rotate(py_img, 90)

            #TODO move to correct position
            position = (0,0)
            pos = self.sim_surface.vec2pix(position)
            self.screen.blit(
                py_img,
                (pos[0] - py_img.get_width() / 2, pos[1] - py_img.get_height()/2),
            )

        pygame.display.flip()


    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
                # paused = True
                # while paused:
                #     pygame.time.wait(1000)
                #     for event in pygame.event.get():
                #         if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                #             paused = False



def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def calc_grid_map_config(ox, oy, xy_resolution):
    """
    Calculates the size, and the maximum distances according to the the
    measurement center
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def init_flood_fill(center_point, obstacle_points, xy_points, min_coord, xy_resolution):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5
    for x, y in zip(ox, oy):
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return occupancy_map


def flood_fill(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))


def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    # min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
    x_w = 200
    y_w = 200
    # print(x_w)
    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
    occupancy_map = np.ones((x_w, y_w)) / 2
    # center_x = int(round(-min_x / xy_resolution))  # center x coordinate of the grid map
    # center_y = int(round(-min_y / xy_resolution))  # center y coordinate of the grid map
    center_x = int(x_w / 2)
    center_y = int(y_w / 2)

    min_x = -int(x_w/2) * xy_resolution
    min_y = -int(y_w/2) * xy_resolution

    # occupancy grid computed with bresenham ray casting
    if breshen:
        for x, y in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham(
                (center_x, center_y), (ix, iy)
            )  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                if laser_beam[0] < x_w and laser_beam[1] < y_w and laser_beam[0] >=0 and laser_beam[1] >= 0:
                    occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0
            if ix >= 0 and iy>= 0 and ix<x_w and iy < y_w:
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            # occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            # occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            # occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    # occupancy grid computed with with flood fill
    else:
        occupancy_map = init_flood_fill(
            (center_x, center_y), (ox, oy), (x_w, y_w), (min_x, min_y), xy_resolution
        )
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for x, y in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    return occupancy_map#, min_x, max_x, min_y, max_y, xy_resolution

def update_ray_casting_grid_map(occupancy_map, ox, oy, xy_resolution, x_w, y_w, lidar_position):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    """

    min_x = -int(x_w/2) * xy_resolution
    min_y = -int(y_w/2) * xy_resolution

    # lidar_pos = np.array([0, 0])
    # convert lidar position from world coordinates to map coordinates
    center_x = int(lidar_position[0] / xy_resolution + x_w/2)
    center_y = int(lidar_position[1] / xy_resolution + y_w/2)

    # occupancy grid computed with bresenham ray casting
    for x, y in zip(ox, oy):
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        laser_beams = bresenham(
            (center_x, center_y), (ix, iy)
        )  # line form the lidar to the occupied point
        for laser_beam in laser_beams:
            if laser_beam[0] < x_w and laser_beam[1] < y_w and laser_beam[0] >=0 and laser_beam[1] >= 0:
                occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0
        if ix >= 0 and iy>= 0 and ix<x_w and iy < y_w:
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
        if ix+1 >= 0 and iy>= 0 and ix+1<x_w and iy < y_w:
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
        if ix >= 0 and iy+1>= 0 and ix<x_w and iy+1 < y_w:
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
        if ix+1 >= 0 and iy+1>= 0 and ix+1<x_w and iy+1 < y_w:
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    return occupancy_map

class Mapper(Node):
    def __init__(self):
        super().__init__("mapper")
        self.first = True
        self.start_pose = None

        self.lidar_position = np.array([0, 0])
        self.lidar_heading = 0.0

        self.renderer = Renderer()

        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile=qos_profile_sensor_data
        )
        self.create_subscription(Pose2D, "car6/ground_pose", self.pose_callback, 10)

        self.get_logger().info("Initialized Mapper")

    def pose_callback(self, msg):
        self.lidar_position = np.array([msg.x, msg.y])
        self.lidar_heading = msg.theta

    def scan_callback(self, msg):
        points = []
        angle = msg.angle_min

        # convert angle range to cartesian x y coordinates
        for r in msg.ranges:
            if r > 0:
                points.append(
                    [r * math.cos(angle + np.pi / 2), r * math.sin(angle + np.pi / 2)]
                )
            angle += msg.angle_increment

        # transform scan to frame of vehicle
        points = np.array(points)
        c, s = (
            # -90 deg: need to account for the rotation of the lidar
            np.cos(-self.lidar_heading - np.deg2rad(90)),  
            np.sin(-self.lidar_heading - np.deg2rad(90)),
        )
        R = np.array(((c, -s), (s, c)))

        points = points @ R
        points += self.lidar_position

        # size of the map in pixels
        x_w = 200
        y_w = 200
        xy_resolution = 0.01  # x-y grid resolution
        self.renderer.grid_resolution = xy_resolution

        # ang = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        # dist = msg.ranges
        # ox = np.sin(ang) * dist
        # oy = np.cos(ang) * dist
        ox = points[:, 0]
        oy = points[:, 1]

        if self.first:
            self.occupancy_map = np.ones((x_w, y_w)) / 2
            self.first = False
        else:
            self.occupancy_map = update_ray_casting_grid_map(self.occupancy_map, ox, oy, xy_resolution, x_w, y_w, self.lidar_position)

        self.renderer.image = (self.occupancy_map * 255).astype(np.uint8)

        self.renderer.points = points
        self.renderer.render()

def main(args=None):
    rclpy.init(args=args)

    tracker = Mapper()
    rclpy.spin(tracker)

if __name__ == "__main__":
    main()
