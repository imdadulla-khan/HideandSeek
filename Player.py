from raycast import Raycast
import math
from walls import *
from vision import Vision, point_rotation


class Player():
    def __init__(self):
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = (128, 128)
        self.rect = rect
        self.angle = 0
        self.checkwalls(850)
        self.orientation = 0
        self.movement_type = [1, 3][0]
        self.dist_covered = 0
        self.vision = Vision(45,180,50).get_lines(self.rect.center, self.near_walls, self.orientation)
        self.render = Vision(45,1000,200).get_lines(self.rect.center, self.near_walls, self.angle)


    def move_axis(self, dx, dy):
        # Moving each axis and checking for collisions with wall
        self.move_single_axis(dx, dy)

    def move_single_axis(self, dx, dy):
        # Moving only one

        dx1, dy1 = dx, dy
        self.rect.x += dx
        self.rect.y += dy
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Move right and hitting the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0:  # Moving left and hitting the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0:  # Moving down and hitting the upper side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0:  # Moving up and hitting the bottom side of the wall
                    self.rect.top = wall.rect.bottom
        self.checkwalls(850)
        self.vision = Vision(45,180,50).get_lines(self.rect.center, self.near_walls, self.angle)
        self.render = Vision(45,1000,200).get_lines(self.rect.center, self.near_walls, self.angle)


    def get_state(self):

        head_position = (self.rect.x, self.rect.y)
        wall_info = tuple(self.is_wall_nearby().values())

        # Concatating the state space
        return head_position + wall_info

    def is_wall_nearby(self):

        left, right, up, down = False, False, False, False
        dx = self.rect.x - 32
        dy = self.rect.y - 32
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    left = "True"
                if dx < 0:  # Moving left and hitting the right side of the wall
                    right = "True"
                if dy > 0: # Moving down and hitting the upper side of the wall
                    down = "True"
                if dy < 0:  # Moving up and hitting the bottom side of the wall
                    up = "True"
        return {
            "LEFT": left,
            "RIGHT": right,
            "UP": up,
            "DOWN": down
        }

    def act(self, direction, rotation):
        is_boundary = self.is_wall_nearby()
        if is_boundary[direction]:
            pass
        else:
            self.move(direction, rotation)


    def move(self, direction, rotation):
        if direction == "LEFT":
            self.move_axis(-25, 0)
        if direction == "RIGHT":
            self.move_axis(25, 0)
        if direction == "UP":
            self.move_axis(0, -25)
        if direction == "DOWN":
            self.move_axis(0, 25)

        if rotation == "l_rot":
            self.angle = (self.angle + 1) % 360
            self.orientation = (self.orientation + 1) % 4
            self.vision = Vision(45,180,50).get_lines(self.rect.center, self.near_walls, self.angle)
            Raycast(self).x_or_y()

        if rotation == "r_rot":
            self.angle = (self.angle - 1) % 360
            self.orientation = (self.orientation - 1) % 4
            self.vision = Vision(45,180,50).get_lines(self.rect.center, self.near_walls, self.angle)
            Raycast(self).x_or_y()

    def checkwalls(self, radius):  # Checking for walls from a given radius from the player
        self.near_walls = []
        for wall in walls:
            if math.dist(self.rect.center, wall.rect.center) <= radius:
                self.near_walls.append(wall)
        pass
